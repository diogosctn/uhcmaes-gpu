"""
uhcmaes_gpu.py
==============
UH-CMA-ES para inversão sísmica — porte fiel do UHCMAES.m (MATLAB/Octave),
com avaliação de fitness em batch na GPU (PyTorch).

Diferenças em relação à versão anterior (uhcmaes_gpu.py original):
  * CMA-ES COMPLETO: caminhos de evolução ps/pc, atualização de C
    (rank-one + rank-mu), teste h_sig, re-decomposição espectral periódica
    e CSA (cumulative step-size adaptation) baseado em ||ps||/chiN;
  * modelo direto correto (denormalize -> log -> D -> Aki-Richards por amostra -> W),
    via SeismicPhysicsGPU (ver seismic_physics.py);
  * normalização configurável ('linear', 'standard', 'log');
  * peso de regularização configurável (reg_type), como no MATLAB;
  * critérios de parada configuráveis (relative_error, correlation,
    chi_squared, stagnation, diversity);
  * logging de histórico (CSV + .mat + backup do config), como no MATLAB.

Estrutura do loop (idêntica ao MATLAB):
  1) amostragem arx = x_new + sigma * B * (D .* z)         (batch, GPU)
  2) forward sísmico de todos os candidatos                (batch, GPU)
  3) fitness = misfit médio sob n_samples ruídos + reg     (batch, GPU)
  4) Uncertainty Handling (reavaliação de subconjunto)     (batch, GPU)
  5) atualização CMA-ES (x_new, ps, pc, C, sigma)          (GPU)
"""

import csv
import json
import os
from pathlib import Path
import random
import string
import time
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
from scipy.signal import butter, filtfilt

from .seismic_physics import SeismicPhysicsGPU


# ==========================================================================
# Uncertainty Handling (idêntico ao uncertainty_measurement do MATLAB)
# ==========================================================================
def uncertainty_measurement_gpu(f_old, f_new, idx_reev, theta_uh):
    lam = f_old.shape[0]
    all_vals = torch.cat([f_old, f_new])
    sort_idx = torch.argsort(all_vals)

    ranks = torch.empty(2 * lam, dtype=f_old.dtype, device=f_old.device)
    ranks[sort_idx] = torch.arange(1, 2 * lam + 1, dtype=f_old.dtype, device=f_old.device)

    r1 = ranks[:lam][idx_reev]
    r2 = ranks[lam:][idx_reev]
    rank_delta = r2 - r1 - torch.sign(r2 - r1)
    limit = lam * theta_uh
    mean_rank_change = torch.mean(torch.abs(rank_delta))
    return ((mean_rank_change - limit) / lam).item()


# ==========================================================================
# Peso de regularização (reg_type do MATLAB)
# ==========================================================================
def _reg_weight(reg_type, generation, sigma):
    if reg_type == "5_exp1":
        return 5.0 * np.exp(-0.1 * generation)
    elif reg_type == "5_exp001":
        return 5.0 * np.exp(-0.001 * generation)
    elif reg_type == "1000_exp0001":
        return 1000.0 * np.exp(-0.00001 * generation)
    elif reg_type == "sigma_exp0001":
        return sigma * np.exp(-0.0001 * generation)
    else:
        return 5.0


# ==========================================================================
# Algoritmo principal
# ==========================================================================
def run_uhcmaes_gpu(cfg, mat_data_path, results_folder, device="cuda",
                    dtype=torch.float64, 
                    seed=None, 
                    verbose_every=25
                ):
    """Executa a inversão UH-CMA-ES.

    Parâmetros
    ----------
    cfg : dict
        Configuração (mesmo schema do config.json do MATLAB).
    mat_data_path : str
        Caminho do .mat com Vp, Vs, Rho, TimeSeis, Snear, Smid, Sfar.
    device : str
        'cuda' ou 'cpu' (cai para CPU se CUDA indisponível).
    dtype : torch.dtype
        torch.float64 (padrão, casa com MATLAB) ou torch.float32 (mais rápido).
    results_folder : str
        Pasta base dos resultados (é criado subdiretório com timestamp).
    seed : int ou None
        Semente para reprodutibilidade.
    verbose_every : int
        Período de impressão/gravação (como o mod(generation,25) do MATLAB).

    Retorna
    -------
    dict com histórico, melhor solução (física) e caminho da pasta de resultados.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Dados
    # ------------------------------------------------------------------
    mat = sio.loadmat(mat_data_path)
    vp = mat["Vp"].squeeze()
    vs = mat["Vs"].squeeze()
    rho = mat["Rho"].squeeze()
    time_seis = mat["TimeSeis"].squeeze()
    snear, smid, sfar = mat["Snear"].squeeze(), mat["Smid"].squeeze(), mat["Sfar"].squeeze()

    y_obs_real = np.concatenate([snear, smid, sfar]).astype(np.float64)
    Nd = len(y_obs_real)

    # ------------------------------------------------------------------
    # 2. Física
    # ------------------------------------------------------------------
    physics = SeismicPhysicsGPU(cfg, vp, vs, rho, time_seis, device=device, dtype=dtype)
    N, nm = physics.N, physics.nm
    assert Nd == physics.Nd, (
        f"Dimensão dos dados ({Nd}) != ntheta*(nm-1) ({physics.Nd}). "
        "Verifique se Snear/Smid/Sfar têm nm-1 amostras (convenção SeReM)."
    )
    y_obs = torch.tensor(y_obs_real, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # 3. Prior (Butterworth + filtfilt) e normalização
    # ------------------------------------------------------------------
    b, a = butter(cfg["physics"]["prior_filter_order"], cfg["physics"]["prior_cutoff_freq"])
    vp_prior = filtfilt(b, a, vp)
    vs_prior = filtfilt(b, a, vs)
    rho_prior = filtfilt(b, a, rho)

    x_prior_np = np.concatenate([
        physics.normalize(vp_prior, physics.p_norm["vp"]),
        physics.normalize(vs_prior, physics.p_norm["vs"]),
        physics.normalize(rho_prior, physics.p_norm["rho"]),
    ])
    x_prior = torch.tensor(x_prior_np, dtype=dtype, device=device).unsqueeze(1)  # (N,1)
    x_new = x_prior.clone()

    # ------------------------------------------------------------------
    # 4. Covariância inicial C = kron(sigma0, sigmatime) e decomposição
    # ------------------------------------------------------------------
    dt = physics.dt
    corrlength = cfg["physics"]["correlation_length_factor"] * dt
    t_arr = np.arange(nm) * dt
    trow, tcol = np.meshgrid(t_arr, t_arr, indexing="xy")
    sigmatime = np.exp(-((trow - tcol) / corrlength) ** 2)

    vp_n = physics.normalize(vp, physics.p_norm["vp"])
    vs_n = physics.normalize(vs, physics.p_norm["vs"])
    rho_n = physics.normalize(rho, physics.p_norm["rho"])
    sigma0 = np.cov(np.vstack([vp_n, vs_n, rho_n]))

    # Salvaguarda algébrica (nugget), como no MATLAB
    if np.sum(np.abs(sigma0)) < 1e-8:
        sigma0 = 1e-4 * np.eye(3)
    else:
        sigma0 = sigma0 + 1e-6 * np.eye(3)

    C = torch.tensor(np.kron(sigma0, sigmatime), dtype=dtype, device=device)
    C = torch.triu(C) + torch.triu(C, diagonal=1).T
    eigvals, B = torch.linalg.eigh(C)
    D_sqrt = torch.sqrt(torch.abs(eigvals)).unsqueeze(1)                     # (N,1)
    D_safe = torch.clamp(D_sqrt, min=1e-12)
    inv_sqrt_C = B @ torch.diag_embed(1.0 / D_safe.squeeze(1)) @ B.T

    # ------------------------------------------------------------------
    # 5. Parâmetros CMA-ES (fórmulas padrão do Hansen, como no MATLAB)
    # ------------------------------------------------------------------
    sigma = float(cfg["cmaes"]["sigma_initial"])
    stop_generations = int(cfg["cmaes"]["stop_generations"])
    stop_tol_diversity = float(cfg["cmaes"]["stop_tol_diversity"])
    reg_type = cfg["cmaes"]["reg_type"]

    lam = 4 + int(np.floor(3 * np.log(N)))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / np.sum(w)
    weights = torch.tensor(w, dtype=dtype, device=device).unsqueeze(1)       # (mu,1)
    mu_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))

    cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
    cs = (mu_eff + 2) / (N + mu_eff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2) ** 2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + cs
    chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    pc = torch.zeros(N, 1, dtype=dtype, device=device)
    ps = torch.zeros(N, 1, dtype=dtype, device=device)
    eigeneval = 0

    # ------------------------------------------------------------------
    # 6. Parâmetros UH
    # ------------------------------------------------------------------
    r_lambda = float(cfg["uh"]["r_lambda"])
    theta_uh = float(cfg["uh"]["theta_uh"])
    alpha_t = float(cfg["uh"]["alpha_t"])
    alpha_sigma = float(cfg["uh"]["alpha_sigma"])
    cs_uh = float(cfg["uh"]["cs_uh"])
    t_eval = float(cfg["uh"]["t_eval_initial"])
    t_min = float(cfg["uh"]["t_min"])
    t_max = float(cfg["uh"]["t_max"])
    noise_level = float(cfg["uh"]["noise_level"])
    s_bar = 0.0

    # ------------------------------------------------------------------
    # 7. Logs / pastas (como no MATLAB)
    # ------------------------------------------------------------------
    unique_hash = "".join(random.choices(string.ascii_lowercase, k=4))
    timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{unique_hash}"

    run_folder = (Path(results_folder) / timestamp).as_posix()

    os.makedirs(run_folder, exist_ok=True)
    csv_path = os.path.join(run_folder, "log_execucao.csv")
    mat_path = os.path.join(run_folder, "run_data.mat")
    json_path = os.path.join(run_folder, "config_used.json")
    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Gen", "Fitness", "Sigma", "Samples", "Diversity", "S_bar",
             "RegWeight", "RelError", "Corr", "Chi2"])

    history = {k: [] for k in
               ["fitness", "t_eval", "sigma", "diversity", "reg_weight",
                "relative_error", "correlation", "chi_squared", "solutions"]}
    history["ensamble"] = None

    print(f"Resultados serão salvos na pasta: {run_folder}")
    print(f"Iniciando Inversão Sísmica com UH-CMA-ES (Dimensão: {N}, "
          f"lambda={lam}, device={device}, dtype={dtype})...")

    # ------------------------------------------------------------------
    # 8. Loop principal
    # ------------------------------------------------------------------
    tempo_execucao = 0.0
    generation = 0
    stop_now = False
    best = {"rel_error": np.inf, "x": None, "generation": 0}

    while generation < stop_generations and not stop_now:
        tic = time.perf_counter()
        generation += 1
        n_samples = int(np.round(t_eval))
        reg_weight = _reg_weight(reg_type, generation, sigma)

        # ---- Passo 1: amostragem + forward + fitness (batch GPU) ----
        Z = torch.randn(N, lam, dtype=dtype, device=device)
        arx = x_new + sigma * (B @ (D_sqrt * Z))                    # (N, lam)

        Y_pred = physics.forward_batch(arx)                         # (lam, Nd)

        noise = noise_level * torch.randn(n_samples, Nd, dtype=dtype, device=device)
        diff = Y_pred.unsqueeze(1) - (y_obs.unsqueeze(0) + noise.unsqueeze(0))
        raw_fitness = torch.mean(torch.sum(diff ** 2, dim=-1), dim=-1)   # (lam,)

        prior_term = torch.sum((arx - x_prior) ** 2, dim=0)         # (lam,)
        arfitness_old = raw_fitness + reg_weight * prior_term

        # ---- Passo 2: Uncertainty Handling ----
        lambda_reev = max(1, int(np.floor(r_lambda * lam)))
        idx_reev = torch.randperm(lam, device=device)[:lambda_reev]

        noise2 = noise_level * torch.randn(n_samples, Nd, dtype=dtype, device=device)
        diff2 = Y_pred[idx_reev].unsqueeze(1) - (y_obs.unsqueeze(0) + noise2.unsqueeze(0))
        raw_reev = torch.mean(torch.sum(diff2 ** 2, dim=-1), dim=-1)
        arfitness_new = arfitness_old.clone()
        arfitness_new[idx_reev] = raw_reev + reg_weight * prior_term[idx_reev]

        s_measure = uncertainty_measurement_gpu(arfitness_old, arfitness_new, idx_reev, theta_uh)
        s_bar = (1 - cs_uh) * s_bar + cs_uh * s_measure
        if s_bar > 0:
            if t_eval < t_max:
                t_eval = min(t_eval * alpha_t, t_max)
            else:
                sigma = sigma * alpha_sigma
        elif s_bar < 0:
            t_eval = max(t_eval / alpha_t, t_min)

        arfitness_final = arfitness_old.clone()
        arfitness_final[idx_reev] = (arfitness_old[idx_reev] + arfitness_new[idx_reev]) / 2.0

        # ---- Passo 3: seleção e métricas ----
        arfitness_sorted, arindex = torch.sort(arfitness_final)
        arx_sorted = arx[:, arindex]

        diversity_metric = torch.mean(torch.sqrt(torch.sum((arx - x_new) ** 2, dim=0))).item()

        history["fitness"].append(arfitness_sorted[0].item())
        history["t_eval"].append(t_eval)
        history["sigma"].append(sigma)
        history["diversity"].append(diversity_metric)
        history["reg_weight"].append(reg_weight)

        # Métricas do melhor indivíduo (sem ruído, como no MATLAB)
        Y_best = Y_pred[arindex[0]]
        rel_error = (torch.norm(y_obs - Y_best) / torch.norm(y_obs)).item()
        yc = y_obs - torch.mean(y_obs)
        pc_ = Y_best - torch.mean(Y_best)
        correlation = (torch.sum(yc * pc_)
                       / (torch.norm(yc) * torch.norm(pc_) + 1e-30)).item()
        chi_squared = torch.mean((y_obs - Y_best) ** 2).item() / noise_level ** 2

        history["relative_error"].append(rel_error)
        history["correlation"].append(correlation)
        history["chi_squared"].append(chi_squared)

        if rel_error < best["rel_error"]:
            best.update(rel_error=rel_error,
                        x=arx_sorted[:, 0].clone(),
                        generation=generation)

        # ---- Critério de parada configurável ----
        stop_method = cfg["cmaes"]["stop_criteria"]["method"]
        stop_threshold = float(cfg["cmaes"]["stop_criteria"]["threshold"])
        if stop_method == "relative_error" and rel_error < stop_threshold:
            stop_now = True
            print(f">>> Parada: Erro Relativo ({rel_error:.4f}) < Tol ({stop_threshold:.4f})")
        elif stop_method == "correlation" and correlation > stop_threshold:
            stop_now = True
            print(f">>> Parada: Correlação ({correlation:.4f}) > Tol ({stop_threshold:.4f})")
        elif stop_method == "chi_squared" and chi_squared < stop_threshold:
            stop_now = True
            print(f">>> Parada: Qui-Quadrado ({chi_squared:.4f}) < Tol ({stop_threshold:.4f})")
        elif stop_method == "stagnation":
            window = int(cfg["cmaes"]["stop_criteria"]["stagnation_window"])
            if generation > window:
                prev_fit = history["fitness"][-window - 1]
                curr_fit = history["fitness"][-1]
                improvement = abs(prev_fit - curr_fit) / (abs(prev_fit) + 1e-30)
                if improvement < stop_threshold:
                    stop_now = True
                    print(f">>> Parada: Estagnação (Melhoria {improvement:.2e}) < Tol ({stop_threshold:.2e})")
        elif stop_method == "diversity" and diversity_metric < stop_tol_diversity:
            stop_now = True
            print(f">>> Parada: Diversidade ({diversity_metric:.4f}) < Tol ({stop_tol_diversity:.4f})")

        if stop_now:
            tempo_execucao += time.perf_counter() - tic
            break

        # ---- Passo 4: atualização CMA-ES COMPLETA ----
        x_old = x_new.clone()
        x_new = arx_sorted[:, :mu] @ weights                        # (N,1)

        # caminho de evolução sigma (CSA)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * \
            inv_sqrt_C @ (x_new - x_old) / sigma
        ps_norm = torch.norm(ps).item()
        h_sig = (ps_norm / np.sqrt(1 - (1 - cs) ** (2 * generation)) / chiN) < \
            (1.4 + 2 / (N + 1))

        # caminho de evolução da covariância
        pc = (1 - cc) * pc + (float(h_sig) * np.sqrt(cc * (2 - cc) * mu_eff)) * \
            (x_new - x_old) / sigma

        artmp = (arx_sorted[:, :mu] - x_old) / sigma                # (N, mu)
        C = ((1 - c1 - c_mu) * C
             + c1 * (pc @ pc.T + (1 - float(h_sig)) * cc * (2 - cc) * C)
             + c_mu * (artmp * weights.T) @ artmp.T)

        sigma = sigma * np.exp((cs / damps) * (ps_norm / chiN - 1))

        # re-decomposição espectral periódica (como no MATLAB)
        if generation - eigeneval > lam / (c1 + c_mu) / N / 10:
            eigeneval = generation
            C = torch.triu(C) + torch.triu(C, diagonal=1).T
            eigvals, B = torch.linalg.eigh(C)
            D_sqrt = torch.sqrt(torch.abs(eigvals)).unsqueeze(1)
            D_safe = torch.clamp(D_sqrt, min=1e-12)
            inv_sqrt_C = B @ torch.diag_embed(1.0 / D_safe.squeeze(1)) @ B.T

        tempo_execucao += time.perf_counter() - tic

        # ---- I/O periódico (fora do cronômetro, como no MATLAB) ----
        if generation % verbose_every == 0:
            x_phys = _denorm_vector(physics, x_new.squeeze(1))
            history["solutions"].append(x_phys.cpu().numpy())
            history["ensamble"] = _denorm_batch(physics, arx).cpu().numpy()
            print(f"Gen {generation}: Fit={arfitness_sorted[0].item():.2e} | "
                  f"Sigma={sigma:.2f} | Div={diversity_metric:.2f} | "
                  f"RelErr={rel_error:.4f} | Corr={correlation:.4f} | Chi2={chi_squared:.2f}")
            _append_csv(csv_path, generation, arfitness_sorted[0].item(), sigma,
                        int(np.round(t_eval)), diversity_metric, s_bar, reg_weight,
                        rel_error, correlation, chi_squared)
            _save_mat(mat_path, history, x_phys.cpu().numpy(), cfg, tempo_execucao)

    # ------------------------------------------------------------------
    # 9. Salvamento final
    # ------------------------------------------------------------------
    x_phys_final = _denorm_vector(physics, x_new.squeeze(1))
    if generation % verbose_every != 0:
        history["solutions"].append(x_phys_final.cpu().numpy())
        history["ensamble"] = _denorm_batch(physics, arx).cpu().numpy()
        _append_csv(csv_path, generation, arfitness_sorted[0].item(), sigma,
                    int(np.round(t_eval)), diversity_metric, s_bar, reg_weight,
                    rel_error, correlation, chi_squared)
    _save_mat(mat_path, history, x_phys_final.cpu().numpy(), cfg, tempo_execucao)

    print(f"Execução concluída na geração {generation} em {tempo_execucao:.2f} segundos.")

    return {
        "history": history,
        "best": {"rel_error": best["rel_error"],
                 "x_phys": _denorm_vector(physics, best["x"]).cpu().numpy()
                 if best["x"] is not None else None,
                 "generation": best["generation"]},
        "x_final_phys": x_phys_final.cpu().numpy(),
        "run_folder": run_folder,
        "generations": generation,
        "time_elapsed": tempo_execucao,
    }


# ==========================================================================
# Utilitários
# ==========================================================================
def _denorm_vector(physics, x):
    """(N,) normalizado -> (N,) físico [Vp; Vs; Rho]."""
    vp, vs, rho = physics.denormalize_batch(x)
    return torch.cat([vp, vs, rho], dim=0)


def _denorm_batch(physics, X):
    """(N, lam) normalizado -> (N, lam) físico."""
    vp, vs, rho = physics.denormalize_batch(X)
    return torch.cat([vp, vs, rho], dim=0)


def _append_csv(csv_path, gen, fit, sigma, samples, div, s_bar, reg_w, rel, corr, chi2):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [gen, f"{fit:.6e}", f"{sigma:.6f}", samples, f"{div:.6f}",
             f"{s_bar:.6f}", f"{reg_w:.6f}", f"{rel:.6f}", f"{corr:.6f}", f"{chi2:.6f}"])


def _save_mat(mat_path, history, x_phys, cfg, tempo):
    hist = {k: (np.array(v) if isinstance(v, list) else v) for k, v in history.items()}
    hist["time_elapsed"] = tempo
    sio.savemat(mat_path, {"history": hist, "x_new": x_phys, "cfg": json.dumps(cfg)})


# ==========================================================================
# Execução direta (exemplo)
# ==========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UH-CMA-ES GPU para inversão sísmica")
    parser.add_argument("--config", default="config.json", help="caminho do config.json")
    parser.add_argument("--data", default=None, help="caminho do .mat (sobrescreve o config)")
    parser.add_argument("--results_folder", default="Results_UHCMAES_py", help="Diretório para armazenar resultado")
    parser.add_argument("--device", default="cuda", help="'cuda' ou 'cpu'")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    data_path = args.data
    if data_path is None:
        data_path = os.path.join(cfg["files"]["data_folder"], cfg["files"]["input_filename"])

    run_uhcmaes_gpu(cfg, data_path, args.results_folder, device=args.device,
                    dtype=getattr(torch, args.dtype), seed=args.seed)
