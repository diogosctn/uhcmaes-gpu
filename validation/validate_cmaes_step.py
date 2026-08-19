"""
validate_cmaes_step.py
======================
Valida a MECÂNICA do CMA-ES (ps, pc, C, sigma/CSA) da versão torch contra
uma referência NumPy que reproduz linha a linha o UHCMAES.m, usando os
mesmos números aleatórios (Z e fitness pré-fixados).
"""
import numpy as np
import torch


def cmaes_step_numpy(x_new, sigma, B, Dv, inv_sqrt_C, C, pc, ps, Z, fitness,
                     weights, mu, mu_eff, cc, cs, c1, c_mu, damps, chiN, generation):
    """Uma geração do CMA-ES, linha a linha como o MATLAB (NumPy)."""
    N = len(x_new)
    lam = Z.shape[1]
    arx = x_new[:, None] + sigma * (B @ (Dv[:, None] * Z))

    arindex = np.argsort(fitness)
    arx_sorted = arx[:, arindex]

    x_old = x_new.copy()
    x_new = arx_sorted[:, :mu] @ weights

    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * \
        inv_sqrt_C @ (x_new - x_old) / sigma
    h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * generation)) / chiN) < \
        (1.4 + 2 / (N + 1))
    pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x_old) / sigma

    artmp = (1 / sigma) * (arx_sorted[:, :mu] - x_old[:, None])
    C = ((1 - c1 - c_mu) * C
         + c1 * (np.outer(pc, pc) + (1 - h_sig) * cc * (2 - cc) * C)
         + c_mu * artmp @ np.diag(weights) @ artmp.T)

    sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
    return x_new, sigma, C, pc, ps, arx


def cmaes_step_torch(x_new, sigma, B, Dv, inv_sqrt_C, C, pc, ps, Z, fitness,
                     weights, mu, mu_eff, cc, cs, c1, c_mu, damps, chiN, generation):
    """Uma geração do CMA-ES, como implementado no uhcmaes_gpu.py (torch)."""
    dt = torch.float64
    N = x_new.shape[0]
    x_new_t = torch.tensor(x_new, dtype=dt).unsqueeze(1)
    B_t = torch.tensor(B, dtype=dt)
    Dv_t = torch.tensor(Dv, dtype=dt).unsqueeze(1)
    inv_t = torch.tensor(inv_sqrt_C, dtype=dt)
    C_t = torch.tensor(C, dtype=dt)
    pc_t = torch.tensor(pc, dtype=dt).unsqueeze(1)
    ps_t = torch.tensor(ps, dtype=dt).unsqueeze(1)
    Z_t = torch.tensor(Z, dtype=dt)
    w_t = torch.tensor(weights, dtype=dt).unsqueeze(1)
    fit_t = torch.tensor(fitness, dtype=dt)

    arx = x_new_t + sigma * (B_t @ (Dv_t * Z_t))

    _, arindex = torch.sort(fit_t)
    arx_sorted = arx[:, arindex]

    x_old = x_new_t.clone()
    x_new_t = arx_sorted[:, :mu] @ w_t

    ps_t = (1 - cs) * ps_t + np.sqrt(cs * (2 - cs) * mu_eff) * \
        inv_t @ (x_new_t - x_old) / sigma
    ps_norm = torch.norm(ps_t).item()
    h_sig = (ps_norm / np.sqrt(1 - (1 - cs) ** (2 * generation)) / chiN) < \
        (1.4 + 2 / (N + 1))
    pc_t = (1 - cc) * pc_t + (float(h_sig) * np.sqrt(cc * (2 - cc) * mu_eff)) * \
        (x_new_t - x_old) / sigma

    artmp = (arx_sorted[:, :mu] - x_old) / sigma
    C_t = ((1 - c1 - c_mu) * C_t
           + c1 * (pc_t @ pc_t.T + (1 - float(h_sig)) * cc * (2 - cc) * C_t)
           + c_mu * (artmp * w_t.T) @ artmp.T)

    sigma = sigma * np.exp((cs / damps) * (ps_norm / chiN - 1))
    return (x_new_t.squeeze(1).numpy(), sigma, C_t.numpy(),
            pc_t.squeeze(1).numpy(), ps_t.squeeze(1).numpy(), arx.numpy())


def main():
    rng = np.random.default_rng(7)
    N, lam = 24, 8
    mu = lam // 2

    # parâmetros CMA-ES
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    mu_eff = w.sum() ** 2 / (w ** 2).sum()
    cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
    cs = (mu_eff + 2) / (N + mu_eff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2) ** 2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + cs
    chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    # estado aleatório (C SPD)
    M = rng.standard_normal((N, N))
    C = M @ M.T + N * np.eye(N)
    evals, B = np.linalg.eigh(C)
    Dv = np.sqrt(np.abs(evals))
    D_safe = np.clip(Dv, 1e-12, None)
    inv_sqrt_C = B @ np.diag(1 / D_safe) @ B.T

    x_new = rng.standard_normal(N)
    pc = rng.standard_normal(N) * 0.01
    ps = rng.standard_normal(N) * 0.01
    sigma = 0.4
    Z = rng.standard_normal((N, lam))
    fitness = rng.random(lam) * 10
    generation = 3

    args = (x_new, sigma, B, Dv, inv_sqrt_C, C, pc, ps, Z, fitness,
            w, mu, mu_eff, cc, cs, c1, c_mu, damps, chiN, generation)

    xn_np, sg_np, C_np, pc_np, ps_np, arx_np = cmaes_step_numpy(*args)
    xn_t, sg_t, C_t, pc_t, ps_t, arx_t = cmaes_step_torch(*args)

    def cmp(name, a, b):
        d = np.max(np.abs(a - b))
        print(f"{name:10s} max|diff| = {d:.3e}")
        return d

    print("Equivalência CMA-ES (NumPy/MATLAB vs torch):")
    cmp("arx", arx_np, arx_t)
    cmp("x_new", xn_np, xn_t)
    cmp("ps", ps_np, ps_t)
    cmp("pc", pc_np, pc_t)
    cmp("C", C_np, C_t)
    cmp("sigma", np.array([sg_np]), np.array([sg_t]))

    tol = 1e-12
    ok = all([
        np.max(np.abs(arx_np - arx_t)) < tol,
        np.max(np.abs(xn_np - xn_t)) < tol,
        np.max(np.abs(ps_np - ps_t)) < tol,
        np.max(np.abs(pc_np - pc_t)) < tol,
        np.max(np.abs(C_np - C_t)) < tol,
        abs(sg_np - sg_t) < tol,
    ])
    print("\nCMA-ES EQUIVALENTE" if ok else "\nCMA-ES DIVERGENTE")
    assert ok


if __name__ == "__main__":
    main()
