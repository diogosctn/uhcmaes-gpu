"""
compare_runs.py — Comparação cruzada MATLAB (out_matlab.mat) vs Python (out_python.mat).

Gera um relatório numérico e figuras comparando:
  1) sísmico do PRIOR (isola o modelo direto / forward);
  2) perfis do prior (Vp/Vs/Rho após filtro Butterworth);
  3) sísmico do melhor indivíduo da 1ª geração (escala/distribuição);
  4) trajetórias de convergência (fitness, sigma, rel_error, correlation);
  5) solução final (Vp/Vs/Rho invertidos).

Uso:
    python3 compare_runs.py
Saídas:
    compare_report.txt, compare_fig1_prior.png, compare_fig2_convergence.png,
    compare_fig3_solution.png
"""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return m


def rel_err(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = min(len(a), len(b))
    return np.corrcoef(a[:n], b[:n])[0, 1]


def hist_to_dict(H):
    return {k: np.atleast_1d(getattr(H, k)) for k in
            ["gen", "fitness", "sigma", "rel_error", "correlation", "t_eval", "diversity"]}


def main():
    M = load("out_matlab.mat")
    P = load("out_python.mat")

    Hm = hist_to_dict(M["HIST"])
    Hp = hist_to_dict(P["HIST"])

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 74)
    log("COMPARAÇÃO CRUZADA  MATLAB (UHCMAES.m)  vs  PYTHON (uhcmaes_gpu.py)")
    log("=" * 74)
    log(f"seed (ambos)            : {int(M['seed'])} / {int(P['seed'])}")
    log(f"N (dimensão)            : {int(M['N'])} / {int(P['N'])}")
    log(f"lambda                  : {int(M['lambda'])} / {int(P['lambda'])}")
    log(f"Nd (dados)              : {int(M['Nd'])} / {int(P['Nd'])}")
    log(f"nm (amostras)           : {int(M['nm'])} / {int(P['nm'])}")
    log("")

    # ---- 1. sísmico do prior (isola o forward) ----
    log("-" * 74)
    log("1) SÍSMICO DO PRIOR  (isola o modelo direto)")
    log("-" * 74)
    e_prior = rel_err(P["Y_prior"], M["Y_prior"])
    c_prior = corr(P["Y_prior"], M["Y_prior"])
    log(f"   rel_err(Y_prior_py, Y_prior_mat) = {e_prior:.3e}")
    log(f"   correlação                       = {c_prior:.6f}")
    veredito_prior = "IDÊNTICOS (forward correto)" if e_prior < 1e-6 else \
        ("PRÓXIMOS" if e_prior < 1e-2 else "DIVERGENTES (verificar forward)")
    log(f"   => {veredito_prior}")
    log("")

    # ---- 2. perfis do prior ----
    log("-" * 74)
    log("2) PERFIS DO PRIOR (Vp/Vs/Rho após Butterworth)")
    log("-" * 74)
    for name in ["VpPrior", "VsPrior", "RhoPrior"]:
        e = rel_err(P[name], M[name])
        log(f"   rel_err({name:9s}) = {e:.3e}   corr={corr(P[name], M[name]):.6f}")
    log("")

    # ---- 3. melhor indivíduo da 1ª geração ----
    log("-" * 74)
    log("3) SÍSMICO DO MELHOR INDIVÍDUO — 1ª GERAÇÃO")
    log("-" * 74)
    log("   (ATENÇÃO: os geradores aleatórios do MATLAB e do NumPy/PyTorch são")
    log("    diferentes, logo o 'melhor' da 1ª geração NÃO é o mesmo indivíduo.")
    log("    Compara-se apenas a escala/distribuição do sísmico.)")
    rms_m = np.sqrt(np.mean(M["Ybest_first"].ravel()**2))
    rms_p = np.sqrt(np.mean(P["Ybest_first"].ravel()**2))
    log(f"   RMS(Ybest_mat) = {rms_m:.4f}   RMS(Ybest_py) = {rms_p:.4f}")
    log(f"   correlação     = {corr(P['Ybest_first'], M['Ybest_first']):.4f} (não precisa ser 1)")
    log("")

    # ---- 4. trajetórias de convergência ----
    log("-" * 74)
    log("4) TRAJETÓRIAS DE CONVERGÊNCIA")
    log("-" * 74)
    log(f"   gerações executadas : MATLAB={len(Hm['gen'])}  Python={len(Hp['gen'])}")
    log(f"   melhor rel_error    : MATLAB={np.min(Hm['rel_error']):.4f}  "
          f"Python={np.min(Hp['rel_error']):.4f}")
    log(f"   rel_error final     : MATLAB={Hm['rel_error'][-1]:.4f}  "
          f"Python={Hp['rel_error'][-1]:.4f}")
    log(f"   melhor correlação   : MATLAB={np.max(Hm['correlation']):.4f}  "
          f"Python={np.max(Hp['correlation']):.4f}")
    log(f"   sigma final         : MATLAB={Hm['sigma'][-1]:.4f}  Python={Hp['sigma'][-1]:.4f}")
    log("")

    # ---- 5. solução final ----
    log("-" * 74)
    log("5) SOLUÇÃO FINAL (Vp/Vs/Rho invertidos)")
    log("-" * 74)
    nm = int(M["nm"])
    xm, xp = M["x_phys_final"].ravel(), P["x_phys_final"].ravel()
    for i, name in enumerate(["Vp", "Vs", "Rho"]):
        a, b = xp[i*nm:(i+1)*nm], xm[i*nm:(i+1)*nm]
        log(f"   rel_err({name:4s}) = {rel_err(a, b):.3e}   corr={corr(a, b):.4f}")
    log("")
    log("=" * 74)
    log("INTERPRETAÇÃO")
    log("=" * 74)
    log("• Se (1) for idêntico, o MODELO DIRETO está correto nos dois ambientes.")
    log("• As trajetórias (4) NÃO serão idênticas ponto a ponto, pois os geradores")
    log("  aleatórios diferem; o esperado é que CONVERJAM para faixas semelhantes")
    log("  de rel_error/correlação, validando a equivalência estatística do método.")
    log("• A solução final (5) deve ter correlação alta (perfis semelhantes), ainda")
    log("  que não bit-idêntica.")

    with open("compare_report.txt", "w") as f:
        f.write("\n".join(lines))

    # ===================== FIGURAS =====================
    # Fig 1: prior
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    nm1 = int(M["nm"]) - 1
    for i, ttl in enumerate(["Snear (10°)", "Smid (20°)", "Sfar (30°)"]):
        sl = slice(i*nm1, (i+1)*nm1)
        axes[i].plot(M["Y_prior"].ravel()[sl], "k-", lw=1.5, label="MATLAB")
        axes[i].plot(P["Y_prior"].ravel()[sl], "r--", lw=1.2, label="Python")
        axes[i].set_title(ttl); axes[i].legend(fontsize=8); axes[i].grid(alpha=0.3)
    fig.suptitle(f"Sísmico do PRIOR  (rel_err={e_prior:.2e})")
    fig.tight_layout(); fig.savefig("compare_fig1_prior.png", dpi=130); plt.close(fig)

    # Fig 2: convergência
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.semilogy(Hm["gen"], Hm["fitness"], "k-", label="MATLAB")
    ax.semilogy(Hp["gen"], Hp["fitness"], "r--", label="Python")
    ax.set_title("Fitness (melhor)"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(Hm["gen"], Hm["sigma"], "k-", label="MATLAB")
    ax.plot(Hp["gen"], Hp["sigma"], "r--", label="Python")
    ax.set_title("Sigma"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(Hm["gen"], Hm["rel_error"], "k-", label="MATLAB")
    ax.plot(Hp["gen"], Hp["rel_error"], "r--", label="Python")
    ax.set_title("Erro relativo"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(Hm["gen"], Hm["correlation"], "k-", label="MATLAB")
    ax.plot(Hp["gen"], Hp["correlation"], "r--", label="Python")
    ax.set_title("Correlação"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Trajetórias de convergência (geradores aleatórios distintos)")
    fig.tight_layout(); fig.savefig("compare_fig2_convergence.png", dpi=130); plt.close(fig)

    # Fig 3: solução final
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(["Vp", "Vs", "Rho"]):
        a, b = xp[i*nm:(i+1)*nm], xm[i*nm:(i+1)*nm]
        axes[i].plot(b, "k-", lw=1.5, label="MATLAB")
        axes[i].plot(a, "r--", lw=1.2, label="Python")
        axes[i].set_title(f"{name} invertido"); axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3); axes[i].invert_xaxis()
    fig.suptitle("Solução final (perfis invertidos)")
    fig.tight_layout(); fig.savefig("compare_fig3_solution.png", dpi=130); plt.close(fig)

    print("\nFiguras salvas: compare_fig1_prior.png, compare_fig2_convergence.png, "
          "compare_fig3_solution.png")
    print("Relatório salvo: compare_report.txt")


if __name__ == "__main__":
    main()
