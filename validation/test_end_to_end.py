"""
test_end_to_end.py
==================
Teste ponta-a-ponta do UH-CMA-ES GPU:
  1) gera um perfil sintético (Vp, Vs, Rho) com nm amostras;
  2) gera o dado observado [Snear; Smid; Sfar] via forward MATLAB (referência);
  3) salva em .mat no formato esperado;
  4) roda run_uhcmaes_gpu e verifica convergência (rel_error decrescendo).
"""
import json
import os

import numpy as np
import scipy.io as sio

from seismic_physics import SeismicPhysicsGPU
from uhcmaes_gpu import run_uhcmaes_gpu
from validate_forward import (ricker_wavelet_matlab, differential_matrix_matlab,
                              wavelet_matrix_matlab, seismic_model_matlab)


def build_synthetic(mat_path, nm=50, seed=1):
    rng = np.random.default_rng(seed)
    dt = 0.001
    t = np.arange(nm) * dt

    # perfis suaves com tendência + camadas
    vp = 3200 + 300 * np.tanh((t - t.mean()) / (t[-1] / 6)) \
        + 80 * np.sin(2 * np.pi * t / (t[-1] / 3))
    vs = 1600 + 180 * np.tanh((t - t.mean()) / (t[-1] / 6)) \
        + 50 * np.sin(2 * np.pi * t / (t[-1] / 3) + 0.5)
    rho = 2350 + 120 * np.tanh((t - t.mean()) / (t[-1] / 7)) \
        + 30 * np.sin(2 * np.pi * t / (t[-1] / 4) + 1.0)

    theta = np.array([10.0, 20.0, 30.0])
    freq, ntw = 30.0, 32
    wavelet = ricker_wavelet_matlab(freq, dt, ntw)
    DiffMat = differential_matrix_matlab(nm, 3)
    WaveMat = wavelet_matrix_matlab(wavelet, nm, len(theta))
    seis = seismic_model_matlab(vp, vs, rho, theta, DiffMat, WaveMat, 3)

    ns = nm - 1
    snear, smid, sfar = seis[:ns], seis[ns:2 * ns], seis[2 * ns:]

    sio.savemat(mat_path, {
        "Vp": vp.reshape(-1, 1), "Vs": vs.reshape(-1, 1), "Rho": rho.reshape(-1, 1),
        "TimeSeis": t.reshape(-1, 1),
        "Snear": snear.reshape(-1, 1), "Smid": smid.reshape(-1, 1), "Sfar": sfar.reshape(-1, 1),
    })
    return {"vp": vp, "vs": vs, "rho": rho, "theta": theta, "freq": freq, "ntw": ntw}


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.join(workdir, "synthetic_data.mat")
    meta = build_synthetic(mat_path, nm=50, seed=1)

    cfg = {
        "files": {"input_filename": "synthetic_data.mat", "data_folder": workdir + os.sep},
        "physics": {
            "nvars": 3,
            "theta_angles": "[10 20 30]",
            "wavelet_freq": meta["freq"],
            "wavelet_ntw": meta["ntw"],
            "prior_filter_order": 3,
            "prior_cutoff_freq": 0.08,
            "correlation_length_factor": 5,
            "normalization": {"method": "standard"},
        },
        "cmaes": {
            "sigma_initial": 0.3,
            "stop_generations": 400,
            "stop_tol_diversity": 1e-6,
            "reg_type": "5_exp001",
            "gen_method": "cmaes",
            "stop_criteria": {"method": "relative_error", "threshold": 0.02,
                              "stagnation_window": 60},
        },
        "uh": {
            "r_lambda": 0.3, "theta_uh": 0.2, "alpha_t": 1.2, "alpha_sigma": 0.9,
            "cs_uh": 0.3, "t_eval_initial": 2, "t_min": 1, "t_max": 50,
            "noise_level": 0.005,
        },
    }
    cfg_path = os.path.join(workdir, "config_test.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    out = run_uhcmaes_gpu(cfg, mat_path, device="cpu", seed=0, verbose_every=20)

    print("\n================ RESUMO ================")
    print(f"gerações executadas : {out['generations']}")
    print(f"tempo               : {out['time_elapsed']:.2f} s")
    print(f"melhor rel_error    : {out['best']['rel_error']:.5f} (gen {out['best']['generation']})")
    re = out["history"]["relative_error"]
    print(f"rel_error inicial   : {re[0]:.5f}")
    print(f"rel_error final     : {re[-1]:.5f}")
    print(f"pasta de resultados : {out['run_folder']}")

    best_rel = out["best"]["rel_error"]
    assert best_rel < 0.5 * re[0], (
        f"ERRO: melhor rel_error ({best_rel:.4f}) não melhorou significativamente "
        f"em relação ao inicial ({re[0]:.4f}).")
    print(f"\nSUCESSO: a inversão convergiu (melhor rel_error {best_rel:.4f} << "
          f"inicial {re[0]:.4f}).")


if __name__ == "__main__":
    main()
