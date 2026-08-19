"""
seismic_physics.py
==================
Modelo direto sísmico linearizado (Aki-Richards + convolução de wavelet de Ricker),
fiel à implementação MATLAB/SeReM (D. Grana, 2020) utilizada pelo UHCMAES.m,
vetorizado para avaliação em batch na GPU (PyTorch).

Cadeia do modelo direto (idêntica ao SeismicModel.m):

    m    = [log(Vp); log(Vs); log(Rho)]          (3*nm,)
    mder = D @ m                                  (3*(nm-1),)
    Cpp  = A(Vp, Vs) @ mder                       (ntheta*(nm-1),)
    Seis = W @ Cpp                                (ntheta*(nm-1),)  == [Snear; Smid; Sfar]

Convenções reproduzidas a partir do SeReM/SeReMpy:
  * DifferentialMatrix: diferença para trás, descarta a 1ª amostra -> blocos (nm-1) x nm;
  * AkiRichardsCoefficientsMatrix: coeficientes POR AMOSTRA, usando avgVs^2/avgVp^2
    local de cada interface (não uma razão Vp/Vs escalar);
  * WaveletMatrix: convmtx(wavelet, nm-1).T, fatiada a partir de argmax(wavelet);
  * RickerWavelet: w = (1 - 2*pi^2*f^2*t^2) * exp(-pi^2*f^2*t^2),
    t = -dt*round(ntw/2) + dt*(0:ntw-1)  (SEM normalização de amplitude).

A normalização é configurável ('linear', 'standard', 'log') como no MATLAB.
O forward é NÃO-LINEAR no espaço do otimizador (passa por denormalize + log),
mas é avaliado para TODOS os candidatos de uma só vez (batch) na GPU.
"""

import numpy as np
import torch
from scipy.linalg import toeplitz


class SeismicPhysicsGPU:
    """Operador direto sísmico em batch (GPU/CPU) compatível com o UHCMAES.m."""

    def __init__(self, cfg, vp, vs, rho, time_seis, device="cuda", dtype=torch.float64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # float64 por padrão para casar com a precisão do MATLAB; use float32 se quiser mais velocidade
        self.dtype = dtype

        vp = np.asarray(vp, dtype=np.float64).squeeze()
        vs = np.asarray(vs, dtype=np.float64).squeeze()
        rho = np.asarray(rho, dtype=np.float64).squeeze()
        time_seis = np.asarray(time_seis, dtype=np.float64).squeeze()

        self.nm = len(vp)
        self.nvars = int(cfg["physics"]["nvars"])
        self.N = self.nm * self.nvars

        # ---------------- Parâmetros físicos ----------------
        self.dt = float(time_seis[1] - time_seis[0])
        self.theta_angles = self._parse_theta(cfg["physics"]["theta_angles"])
        self.ntheta = len(self.theta_angles)
        self.freq = float(cfg["physics"]["wavelet_freq"])
        self.ntw = int(cfg["physics"]["wavelet_ntw"])
        self.Nd = self.ntheta * (self.nm - 1)  # dimensão de [Snear; Smid; Sfar]

        # ---------------- Normalização (como no MATLAB) ----------------
        self.norm_method = cfg["physics"]["normalization"]["method"]
        self.p_norm = {
            "vp":  self._norm_params(vp),
            "vs":  self._norm_params(vs),
            "rho": self._norm_params(rho),
        }

        # ---------------- Matrizes fixas do operador ----------------
        D_np = self._differential_matrix()                       # 3(nm-1) x 3nm
        W_np = self._wavelet_matrix()                            # ntheta(nm-1) x ntheta(nm-1)
        self.D = torch.tensor(D_np, dtype=self.dtype, device=self.device)
        self.W = torch.tensor(W_np, dtype=self.dtype, device=self.device)

        # Pré-computos para Aki-Richards por amostra (ângulo em radianos)
        th = np.deg2rad(self.theta_angles)
        self._tan2 = torch.tensor(np.tan(th) ** 2, dtype=self.dtype, device=self.device)   # (ntheta,)
        self._sin2 = torch.tensor(np.sin(th) ** 2, dtype=self.dtype, device=self.device)   # (ntheta,)

    # ======================================================================
    # Construção das matrizes (convenções SeReM)
    # ======================================================================
    @staticmethod
    def _parse_theta(theta_raw):
        """Aceita lista, string '[10 20 30]'/'10,20,30' ou escalar (como o MATLAB)."""
        if isinstance(theta_raw, (list, tuple, np.ndarray)):
            arr = np.asarray(theta_raw, dtype=np.float64).ravel()
        elif isinstance(theta_raw, str):
            arr = np.fromstring(theta_raw.strip().strip("[]").replace(",", " "), sep=" ")
        else:
            arr = np.atleast_1d(np.asarray(theta_raw, dtype=np.float64))
        if arr.size == 0:
            raise ValueError("theta_angles vazio ou inválido no config.json")
        return arr

    @staticmethod
    def _norm_params(x):
        return {
            "min": float(np.min(x)), "max": float(np.max(x)),
            "mean": float(np.mean(x)), "std": float(np.std(x)),  # std populacional == std(...,1) do MATLAB
        }

    def _ricker_wavelet(self):
        """RickerWavelet.m do SeReM (sem normalização de amplitude)."""
        tmin = -self.dt * round(self.ntw / 2)
        tw = tmin + self.dt * np.arange(self.ntw)
        w = (1.0 - 2.0 * (np.pi ** 2 * self.freq ** 2) * tw ** 2) * \
            np.exp(-(np.pi ** 2 * self.freq ** 2) * tw ** 2)
        return w

    def _differential_matrix(self):
        """DifferentialMatrix.m: bloco-diagonal, cada bloco (nm-1) x nm (diferença para trás)."""
        nm, nv = self.nm, self.nvars
        I = np.eye(nm)
        B = np.zeros((nm, nm))
        B[1:, :-1] = -np.eye(nm - 1)
        J = (I + B)[1:, :]                       # (nm-1) x nm
        D = np.zeros(((nm - 1) * nv, nm * nv))
        for i in range(nv):
            D[i * (nm - 1):(i + 1) * (nm - 1), i * nm:(i + 1) * nm] = J
        return D

    def _wavelet_matrix(self):
        """WaveletMatrix.m: convmtx(w, nm-1).T fatiada de argmax(w), bloco por ângulo."""
        ns = self.nm - 1
        wavelet = self._ricker_wavelet()
        indmax = int(np.argmax(wavelet))
        # convmtx do MATLAB: toeplitz([w; zeros(ns-1,1)], [w(1), zeros(1,ns-1)])
        # -> (ntw+ns-1) x ns, tal que convmtx(w,ns) @ x = conv(w, x) para x (ns,).
        # A WaveletMatrix do SeReM fatia as linhas [indmax : indmax+ns-1] dessa
        # matriz, o que implementa a convolucao 'same' centrada no pico da wavelet.
        col = np.r_[wavelet, np.zeros(ns - 1)]
        lin = np.r_[wavelet[0], np.zeros(ns - 1)]
        conv_mat = toeplitz(col, lin)                             # (ntw+ns-1) x ns
        wsub = conv_mat[indmax:indmax + ns, :]                    # ns x ns
        W = np.zeros((self.ntheta * ns, self.ntheta * ns))
        for i in range(self.ntheta):
            W[i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] = wsub
        return W

    # ======================================================================
    # Normalização (idêntica às funções normalize_data/denormalize_data do MATLAB)
    # ======================================================================
    def normalize(self, val, p):
        if self.norm_method == "linear":
            denom = p["max"] - p["min"]
            denom = denom if denom != 0 else 1e-6
            return (val - p["min"]) / denom
        elif self.norm_method == "standard":
            denom = p["std"] if p["std"] != 0 else 1e-6
            return (val - p["mean"]) / denom
        elif self.norm_method == "log":
            return np.log(val)
        else:
            denom = p["max"] - p["min"]
            denom = denom if denom != 0 else 1e-6
            return (val - p["min"]) / denom

    def denormalize(self, val_n, p):
        if self.norm_method == "linear":
            return val_n * (p["max"] - p["min"]) + p["min"]
        elif self.norm_method == "standard":
            return val_n * p["std"] + p["mean"]
        elif self.norm_method == "log":
            return torch.exp(val_n) if torch.is_tensor(val_n) else np.exp(val_n)
        else:
            return val_n * (p["max"] - p["min"]) + p["min"]

    def denormalize_batch(self, X):
        """X: (N, lambda) ou (N,) tensor no espaço normalizado -> propriedades físicas.

        Retorna vp, vs, rho com shape (nm, lambda) ou (nm,)."""
        nm = self.nm
        vp_n, vs_n, rho_n = X[:nm], X[nm:2 * nm], X[2 * nm:]
        vp = self.denormalize(vp_n, self.p_norm["vp"])
        vs = self.denormalize(vs_n, self.p_norm["vs"])
        rho = self.denormalize(rho_n, self.p_norm["rho"])
        return vp, vs, rho

    # ======================================================================
    # Modelo direto em batch (fiel ao SeismicModel.m)
    # ======================================================================
    def forward_batch(self, X):
        """Avalia o sísmico sintético de todos os candidatos de uma vez.

        Parâmetros
        ----------
        X : tensor (N, lambda) no espaço NORMALIZADO do otimizador.

        Retorna
        -------
        Y : tensor (lambda, Nd) com Nd = ntheta*(nm-1), ordenado [near; mid; far].
        """
        nm = self.nm
        vp, vs, rho = self.denormalize_batch(X)          # (nm, lambda) cada
        vp = torch.clamp(vp, min=1e-8)                   # protege o log (equivalente ao fallback do MATLAB)
        vs = torch.clamp(vs, min=1e-8)
        rho = torch.clamp(rho, min=1e-8)

        # m = [log(Vp); log(Vs); log(Rho)] -> (3nm, lambda)
        m = torch.cat([torch.log(vp), torch.log(vs), torch.log(rho)], dim=0)
        mder = self.D @ m                                # (3(nm-1), lambda)
        dvp, dvs, drho = torch.chunk(mder, 3, dim=0)     # (nm-1, lambda) cada

        # Aki-Richards POR AMOSTRA (avgVs^2/avgVp^2 local de cada interface)
        avg_vp = 0.5 * (vp[:-1] + vp[1:])                # (nm-1, lambda)
        avg_vs = 0.5 * (vs[:-1] + vs[1:])
        ratio = (avg_vs ** 2) / (avg_vp ** 2)            # (nm-1, lambda)

        tan2 = self._tan2.view(-1, 1, 1)                 # (ntheta, 1, 1)
        sin2 = self._sin2.view(-1, 1, 1)                 # (ntheta, 1, 1)
        r = ratio.unsqueeze(0)                           # (1, nm-1, lambda)

        cp = 0.5 * (1.0 + tan2)                          # (ntheta, 1, 1)
        cs = -4.0 * sin2 * r                             # (ntheta, nm-1, lambda)
        cr = 0.5 - 2.0 * sin2 * r                        # (ntheta, nm-1, lambda)

        cpp = (cp * dvp.unsqueeze(0)
               + cs * dvs.unsqueeze(0)
               + cr * drho.unsqueeze(0))                 # (ntheta, nm-1, lambda)
        cpp = cpp.reshape(self.ntheta * (nm - 1), -1)    # (Nd, lambda)

        return (self.W @ cpp).T                          # (lambda, Nd)

    def forward_single(self, x):
        """Conveniência para um único candidato (N,) -> (Nd,)."""
        return self.forward_batch(x.unsqueeze(1)).squeeze(0)
