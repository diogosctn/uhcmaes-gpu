"""
validate_forward.py
===================
Valida o novo SeismicPhysicsGPU contra uma referência NumPy que reproduz
EXATAMENTE o SeismicModel.m + AkiRichardsCoefficientsMatrix.m +
DifferentialMatrix.m + WaveletMatrix.m + RickerWavelet.m do SeReM/MATLAB.

Também compara com o forward "antigo" (bugado): G fixo, Vp/Vs escalar, sem log.
"""
import numpy as np
import torch
from scipy.linalg import toeplitz

from seismic_physics import SeismicPhysicsGPU


# --------------------------------------------------------------------------
# Referência NumPy fiel ao MATLAB (linha a linha)
# --------------------------------------------------------------------------
def ricker_wavelet_matlab(freq, dt, ntw):
    tmin = -dt * round(ntw / 2)
    tw = tmin + dt * np.arange(ntw)
    w = (1 - 2 * (np.pi**2 * freq**2) * tw**2) * np.exp(-(np.pi**2 * freq**2) * tw**2)
    return w


def differential_matrix_matlab(nt, nv):
    I = np.eye(nt)
    B = np.zeros((nt, nt))
    B[1:, :-1] = -np.eye(nt - 1)
    J = (I + B)[1:, :]
    D = np.zeros(((nt - 1) * nv, nt * nv))
    for i in range(nv):
        D[i * (nt - 1):(i + 1) * (nt - 1), i * nt:(i + 1) * nt] = J
    return D


def convmtx_matlab(w, ns):
    # convmtx do MATLAB: C = toeplitz([w; zeros(ns-1,1)], [w(1), zeros(1,ns-1)])
    # primeiro argumento = primeira COLUNA, segundo = primeira LINHA.
    # Resultado: (len(w)+ns-1) x ns, tal que C @ x = conv(w, x) para x de tamanho ns.
    col = np.r_[w, np.zeros(ns - 1)]
    lin = np.r_[w[0], np.zeros(ns - 1)]
    return toeplitz(col, lin)


def wavelet_matrix_matlab(wavelet, nsamples, ntheta):
    ns = nsamples - 1
    W = np.zeros((ntheta * ns, ntheta * ns))
    indmax = int(np.argmax(wavelet))
    conv_mat = convmtx_matlab(wavelet, ns)          # (ntw+ns-1) x ns
    wsub = conv_mat[indmax:indmax + ns, :]          # ns x ns (convolucao 'same')
    for i in range(ntheta):
        W[i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] = wsub
    return W


def aki_richards_matlab(Vp, Vs, theta_deg, nv):
    nsamples = len(Vp)
    ntheta = len(theta_deg)
    A = np.zeros(((nsamples - 1) * ntheta, nv * (nsamples - 1)))
    avgVp = 0.5 * (Vp[:-1] + Vp[1:])
    avgVs = 0.5 * (Vs[:-1] + Vs[1:])
    for i in range(ntheta):
        th = theta_deg[i] * np.pi / 180
        cp = 0.5 * (1 + np.tan(th) ** 2) * np.ones(nsamples - 1)
        cs = -4 * avgVs**2 / avgVp**2 * np.sin(th) ** 2
        cr = 0.5 * (1 - 4 * avgVs**2 / avgVp**2 * np.sin(th) ** 2)
        A[i * (nsamples - 1):(i + 1) * (nsamples - 1), :] = \
            np.hstack([np.diag(cp), np.diag(cs), np.diag(cr)])
    return A


def seismic_model_matlab(Vp, Vs, Rho, theta_deg, DiffMat, WaveMat, nv):
    m = np.concatenate([np.log(Vp), np.log(Vs), np.log(Rho)])
    mder = DiffMat @ m
    A = aki_richards_matlab(Vp, Vs, theta_deg, nv)
    Cpp = A @ mder
    return WaveMat @ Cpp


# --------------------------------------------------------------------------
# Forward "antigo" (bugado) para comparação
# --------------------------------------------------------------------------
def forward_old_buggy(vp_n, vs_n, rho_n, p_norm, G):
    x = np.concatenate([vp_n, vs_n, rho_n])
    return G @ x  # linear no espaço normalizado, sem log, Vp/Vs escalar


# --------------------------------------------------------------------------
# Teste principal
# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # Perfil sintético suave (nm amostras)
    nm = 60
    t = np.arange(nm) * 0.001  # dt = 1 ms
    dt = 0.001
    vp = 3000 + 400 * np.sin(2 * np.pi * t / t[-1] * 1.5) + 50 * rng.standard_normal(nm).cumsum() * 0.1
    vs = 1500 + 250 * np.sin(2 * np.pi * t / t[-1] * 1.5 + 0.4) + 30 * rng.standard_normal(nm).cumsum() * 0.1
    rho = 2400 + 150 * np.sin(2 * np.pi * t / t[-1] * 1.2 + 0.8) + 10 * rng.standard_normal(nm).cumsum() * 0.1
    vp, vs, rho = np.abs(vp) + 1000, np.abs(vs) + 500, np.abs(rho) + 1000

    theta = np.array([10.0, 20.0, 30.0])
    nvars = 3
    freq, ntw = 30.0, 64

    cfg = {
        "physics": {
            "nvars": nvars,
            "theta_angles": "[10 20 30]",
            "wavelet_freq": freq,
            "wavelet_ntw": ntw,
            "normalization": {"method": "standard"},
            "prior_filter_order": 3,
            "prior_cutoff_freq": 0.1,
            "correlation_length_factor": 5,
        }
    }

    time_seis = np.arange(nm) * dt

    # ---------- Referência MATLAB ----------
    wavelet = ricker_wavelet_matlab(freq, dt, ntw)
    DiffMat = differential_matrix_matlab(nm, nvars)
    WaveMat = wavelet_matrix_matlab(wavelet, nm, len(theta))
    ref = seismic_model_matlab(vp, vs, rho, theta, DiffMat, WaveMat, nvars)

    # ---------- Novo (GPU/torch) ----------
    physics = SeismicPhysicsGPU(cfg, vp, vs, rho, time_seis, device="cpu", dtype=torch.float64)
    # normaliza e re-denormaliza para alimentar o forward no espaço do otimizador
    vp_n = physics.normalize(vp, physics.p_norm["vp"])
    vs_n = physics.normalize(vs, physics.p_norm["vs"])
    rho_n = physics.normalize(rho, physics.p_norm["rho"])
    x_n = torch.tensor(np.concatenate([vp_n, vs_n, rho_n]),
                       dtype=torch.float64).unsqueeze(1)
    new = physics.forward_batch(x_n).squeeze(0).numpy()

    # ---------- Antigo (bugado) ----------
    # reconstrói G do código antigo para comparação
    D_sub = np.eye(nm) - np.eye(nm, k=-1); D_sub[0, 0] = 0
    D = np.block([[D_sub, np.zeros((nm, nm)), np.zeros((nm, nm))],
                  [np.zeros((nm, nm)), D_sub, np.zeros((nm, nm))],
                  [np.zeros((nm, nm)), np.zeros((nm, nm)), D_sub]])
    vp_vs_avg = np.mean(vs / vp)
    th = np.radians(theta)
    A_blocks = []
    for k in range(len(th)):
        c1 = 0.5 * (1 + np.tan(th[k])**2)
        c2 = -4 * vp_vs_avg**2 * np.sin(th[k])**2
        c3 = 0.5 - 2 * vp_vs_avg**2 * np.sin(th[k])**2
        A_blocks.append(np.hstack([c1*np.eye(nm), c2*np.eye(nm), c3*np.eye(nm)]))
    A_old = np.vstack(A_blocks)
    wl = (1 - 2*(np.pi*freq*(np.arange(-ntw/2, ntw/2)*dt))**2) * \
         np.exp(-(np.pi*freq*(np.arange(-ntw/2, ntw/2)*dt))**2)
    wl = wl / np.max(np.abs(wl))
    W_sub = np.zeros((nm, nm)); half = len(wl)//2
    for i in range(nm):
        for j in range(len(wl)):
            idx = i - half + j
            if 0 <= idx < nm:
                W_sub[i, idx] = wl[j]
    W_old = np.block([[W_sub if i == j else np.zeros((nm, nm))
                       for j in range(len(th))] for i in range(len(th))])
    G_old = W_old @ A_old @ D
    old = forward_old_buggy(vp_n, vs_n, rho_n, physics.p_norm, G_old)

    # ---------- Métricas ----------
    def metrics(a, b, name):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        err = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)
        corr = np.corrcoef(a, b)[0, 1]
        print(f"{name:28s}  rel_err={err:.3e}   corr={corr:.6f}   "
              f"max_abs_diff={np.max(np.abs(a-b)):.3e}")
        return err, corr

    print(f"nm={nm}, ntheta={len(theta)}, Nd(ref)={len(ref)}, Nd(new)={len(new)}, Nd(old)={len(old)}")
    print("-" * 80)
    metrics(new, ref, "NOVO (GPU) vs MATLAB ref")
    metrics(old, ref, "ANTIGO (bugado) vs MATLAB ref")

    # teste batch: 5 candidatos perturbados
    print("-" * 80)
    X = torch.cat([x_n + 0.05 * torch.randn_like(x_n) for _ in range(5)], dim=1)
    Yb = physics.forward_batch(X).numpy()
    ok = True
    for k in range(5):
        xk = X[:, k].numpy()
        vpk = physics.denormalize(xk[:nm], physics.p_norm["vp"])
        vsk = physics.denormalize(xk[nm:2*nm], physics.p_norm["vs"])
        rhok = physics.denormalize(xk[2*nm:], physics.p_norm["rho"])
        refk = seismic_model_matlab(vpk, vsk, rhok, theta, DiffMat, WaveMat, nvars)
        e = np.linalg.norm(Yb[k] - refk) / np.linalg.norm(refk)
        print(f"candidato {k}: rel_err={e:.3e}")
        ok &= e < 1e-10
    print("-" * 80)
    print("BATCH OK" if ok else "BATCH FALHOU")


if __name__ == "__main__":
    main()
