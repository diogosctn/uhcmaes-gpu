"""
run_python.py — Runner Python do UH-CMA-ES para comparação cruzada.

Roda a MESMA config.json e a MESMA seed do runner MATLAB (run_matlab.m),
exportando os mesmos artefatos para comparação:
  sísmico do prior, sísmico do melhor indivíduo da 1ª geração, histórico
  (fitness, sigma, rel_error, correlation, t_eval, diversity) e solução final.

Uso:
    python3 run_python.py                      # device auto, dtype float64
    python3 run_python.py --device cuda --dtype float32
Grava:
    out_python.mat
"""
import argparse
import json

import numpy as np
import scipy.io as sio
import torch
from scipy.signal import butter, filtfilt

from seismic_physics import SeismicPhysicsGPU
from uhcmaes_gpu import run_uhcmaes_gpu


def load_data(cfg):
    mat = sio.loadmat(cfg["files"]["data_folder"] + cfg["files"]["input_filename"])
    return (mat["Vp"].squeeze(), mat["Vs"].squeeze(), mat["Rho"].squeeze(),
            mat["TimeSeis"].squeeze(),
            np.concatenate([mat["Snear"].squeeze(), mat["Smid"].squeeze(),
                            mat["Sfar"].squeeze()]))


def prior_profiles(cfg, vp, vs, rho):
    b, a = butter(cfg["physics"]["prior_filter_order"],
                  cfg["physics"]["prior_cutoff_freq"])
    return filtfilt(b, a, vp), filtfilt(b, a, vs), filtfilt(b, a, rho)


def prior_seismic(cfg, physics, vp_prior, vs_prior, rho_prior, dtype):
    x_prior_n = torch.tensor(np.concatenate([
        physics.normalize(vp_prior, physics.p_norm["vp"]),
        physics.normalize(vs_prior, physics.p_norm["vs"]),
        physics.normalize(rho_prior, physics.p_norm["rho"]),
    ]), dtype=dtype, device=physics.device).unsqueeze(1)
    Y = physics.forward_batch(x_prior_n).squeeze(0).cpu().numpy()
    return Y, x_prior_n


def first_gen_best_seismic(cfg, physics, x_prior_n, y_obs, seed, dtype):
    """Reexecuta a 1ª geração (mesma seed) e devolve o sísmico do melhor indivíduo.

    NOTA: isto reproduz a amostragem da 1ª geração do run_uhcmaes_gpu. Como os
    geradores de números aleatórios do MATLAB e do NumPy/PyTorch são diferentes,
    o MELHOR indivíduo da 1ª geração não será o mesmo entre os ambientes — este
    artefato serve para comparar a DISTRIBUIÇÃO/escala do sísmico, não valores
    ponto a ponto.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, N, nm, dt = physics.device, physics.N, physics.nm, physics.dt

    corrlength = cfg["physics"]["correlation_length_factor"] * dt
    t_arr = np.arange(nm) * dt
    trow, tcol = np.meshgrid(t_arr, t_arr, indexing="xy")
    sigmatime = np.exp(-((trow - tcol) / corrlength) ** 2)
    # sigma0 a partir dos dados brutos (como no runner principal)
    # (recarregamos para não depender de estado externo)
    mat = sio.loadmat(cfg["files"]["data_folder"] + cfg["files"]["input_filename"])
    vp_n = physics.normalize(mat["Vp"].squeeze(), physics.p_norm["vp"])
    vs_n = physics.normalize(mat["Vs"].squeeze(), physics.p_norm["vs"])
    rho_n = physics.normalize(mat["Rho"].squeeze(), physics.p_norm["rho"])
    sigma0 = np.cov(np.vstack([vp_n, vs_n, rho_n]))
    sigma0 = (1e-4 * np.eye(3)) if np.sum(np.abs(sigma0)) < 1e-8 else sigma0 + 1e-6 * np.eye(3)

    C = torch.tensor(np.kron(sigma0, sigmatime), dtype=dtype, device=device)
    C = torch.triu(C) + torch.triu(C, diagonal=1).T
    eigvals, B = torch.linalg.eigh(C)
    D_sqrt = torch.sqrt(torch.abs(eigvals)).unsqueeze(1)

    sigma = float(cfg["cmaes"]["sigma_initial"])
    lam = 4 + int(np.floor(3 * np.log(N)))
    n_samples = int(np.round(float(cfg["uh"]["t_eval_initial"])))
    noise_level = float(cfg["uh"]["noise_level"])
    y_obs_t = torch.tensor(y_obs, dtype=dtype, device=device)

    Z = torch.randn(N, lam, dtype=dtype, device=device)
    arx = x_prior_n + sigma * (B @ (D_sqrt * Z))
    Y_pred = physics.forward_batch(arx)
    noise = noise_level * torch.randn(n_samples, physics.Nd, dtype=dtype, device=device)
    diff = Y_pred.unsqueeze(1) - (y_obs_t.unsqueeze(0) + noise.unsqueeze(0))
    raw_fitness = torch.mean(torch.sum(diff ** 2, dim=-1), dim=-1)
    reg_weight = 5.0 * np.exp(-0.001) if cfg["cmaes"]["reg_type"] == "5_exp001" else 5.0
    prior_term = torch.sum((arx - x_prior_n) ** 2, dim=0)
    arfitness_old = raw_fitness + reg_weight * prior_term

    _, arindex = torch.sort(arfitness_old)
    return Y_pred[arindex[0]].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    seed = int(cfg.get("compare", {}).get("seed", 42))
    data_path = cfg["files"]["data_folder"] + cfg["files"]["input_filename"]
    dtype = getattr(torch, args.dtype)

    # ---- roda a inversão (mesma seed) ----
    out = run_uhcmaes_gpu(cfg, data_path, device=args.device, dtype=dtype,
                          seed=seed, verbose_every=20)

    # ---- artefatos de comparação ----
    vp, vs, rho, time_seis, y_obs = load_data(cfg)
    physics = SeismicPhysicsGPU(cfg, vp, vs, rho, time_seis,
                                device=args.device, dtype=dtype)
    vp_prior, vs_prior, rho_prior = prior_profiles(cfg, vp, vs, rho)
    Y_prior, x_prior_n = prior_seismic(cfg, physics, vp_prior, vs_prior,
                                       rho_prior, dtype)
    Ybest_first = first_gen_best_seismic(cfg, physics, x_prior_n, y_obs, seed, dtype)

    hist = out["history"]
    HIST = {
        "gen": np.arange(1, len(hist["fitness"]) + 1, dtype=np.float64),
        "fitness": np.asarray(hist["fitness"], dtype=np.float64),
        "sigma": np.asarray(hist["sigma"], dtype=np.float64),
        "rel_error": np.asarray(hist["relative_error"], dtype=np.float64),
        "correlation": np.asarray(hist["correlation"], dtype=np.float64),
        "t_eval": np.asarray(hist["t_eval"], dtype=np.float64),
        "diversity": np.asarray(hist["diversity"], dtype=np.float64),
    }

    sio.savemat("out_python.mat", {
        "Y_prior": Y_prior.reshape(-1, 1),
        "Ybest_first": np.asarray(Ybest_first).reshape(-1, 1),
        "HIST": HIST,
        "x_phys_final": out["x_final_phys"].reshape(-1, 1),
        "VpPrior": vp_prior.reshape(-1, 1),
        "VsPrior": vs_prior.reshape(-1, 1),
        "RhoPrior": rho_prior.reshape(-1, 1),
        "y_obs_real": y_obs.reshape(-1, 1),
        "N": physics.N, "lambda": 4 + int(np.floor(3 * np.log(physics.N))),
        "Nd": physics.Nd, "nm": physics.nm, "seed": seed,
    })
    print(f"PYTHON: salvo out_python.mat (gen final={out['generations']}, "
          f"melhor rel_error={out['best']['rel_error']:.4f})")


if __name__ == "__main__":
    main()
