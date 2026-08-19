# UH-CMA-ES GPU — Inversão Sísmica Paralelizada

Porte para **Python/PyTorch** do algoritmo **UH-CMA-ES** de inversão sísmica
(original em MATLAB/Octave), com avaliação de fitness **em batch na GPU** e
**CMA-ES completo** (adaptação de covariância e de passo). Resolve a inversão
sísmica pré-empilhamento (AVO/AVA) de um traço 1D, estimando perfis de
**Vp, Vs e densidade (Rho)** a partir de três ângulos (near/mid/far), usando um
modelo direto linearizado de **Aki-Richards** com convolução de wavelet de
Ricker (baseado na biblioteca [SeReM](https://github.com/seismicreservoirmodeling/SeReM)
de D. Grana).

## Destaques

- **Modelo direto fiel ao MATLAB**: `denormalizar → log → D → Aki-Richards
  (por amostra, com avgVs²/avgVp² local) → W`, validado contra a referência
  SeReM com erro de ~1e-16 (precisão de máquina).
- **CMA-ES completo**: caminhos de evolução `ps`/`pc`, atualização da matriz de
  covariância `C` (rank-one + rank-μ), teste `h_sig`, re-decomposição espectral
  periódica e **CSA** (cumulative step-size adaptation) baseado em `‖ps‖/chiN`.
- **Uncertainty Handling (UH)** de Hansen et al.: reavaliação de subconjunto,
  métrica de mudança de ranking e ajuste adaptativo de `t_eval`/`sigma`.
- **Paralelização em GPU**: todos os λ candidatos de uma geração são avaliados
  de uma só vez (batch) via PyTorch.
- **Normalização configurável** (`linear`/`standard`/`log`), regularização
  configurável (`reg_type`), 5 critérios de parada e logging (CSV + `.mat` +
  backup do config), no mesmo schema do `config.json` do MATLAB.

## Estrutura

```
uhcmaes-gpu/
├── uhcmaes_gpu/            # pacote principal
│   ├── seismic_physics.py  # modelo direto em batch GPU (fiel ao SeReM)
│   └── uhcmaes_gpu.py      # UH-CMA-ES completo + UH + paradas + logging
├── validation/             # testes de equivalência numérica
│   ├── validate_forward.py     # forward GPU vs referência MATLAB (~1e-16)
│   ├── validate_cmaes_step.py  # 1 geração CMA-ES vs MATLAB (<1e-15)
│   └── test_end_to_end.py      # inversão completa com dado sintético
├── compare/                # comparação cruzada MATLAB vs Python
│   ├── run_matlab.m            # runner MATLAB/Octave (réplica do UHCMAES.m)
│   ├── run_python.py           # runner Python
│   ├── compare_runs.py         # relatório + figuras
│   ├── SeReM/                  # funções SeReM oficiais (.m)
│   └── matlab/                 # UHCMAES.m, SeismicModel.m, AkiRichards...m
├── examples/               # config.json de exemplo + dado real (.mat)
└── docs/
```

## Instalação

```bash
git clone <este-repo>
cd uhcmaes-gpu
pip install torch scipy numpy matplotlib
```

## Uso rápido

```bash
# inversão (mesmo schema de config.json do MATLAB)
python -m uhcmaes_gpu.uhcmaes_gpu \
    --config examples/config.json --device cuda --dtype float64 --seed 42
```

Ou programaticamente:

```python
import json
from uhcmaes_gpu import run_uhcmaes_gpu

cfg = json.load(open("examples/config.json"))
out = run_uhcmaes_gpu(cfg, "examples/data_segy_IL118_XL229.mat",
                      device="cuda", seed=42)
print("melhor rel_error:", out["best"]["rel_error"])
```

## Validação

```bash
cd validation
python validate_forward.py      # forward GPU vs MATLAB  → rel_err ~1e-16
python validate_cmaes_step.py   # passo CMA-ES vs MATLAB → max|diff| < 1e-15
python test_end_to_end.py       # inversão converge (rel_error decrescente)
```

## Comparação cruzada MATLAB vs Python

```bash
cd compare
octave --quiet run_matlab.m          # ou: matlab -batch run_matlab
python run_python.py --device cuda
python compare_runs.py               # gera relatório + 3 figuras
```

Veja `compare/README_compare.md` para detalhes. **Nota**: o
`compare/matlab/AkiRichardsCoefficientsMatrix.m` foi corrigido de
`avgVs.^2. / avgVp.^2` para `avgVs.^2 ./ avgVp.^2` (o espaço fazia o Octave
interpretar `/` como divisão matricial). Aplique a mesma correção ao seu
arquivo MATLAB original.

## Observações

- **dtype**: `torch.float64` (padrão) reproduz a precisão do MATLAB;
  `torch.float32` é mais rápido na GPU.
- **Dimensão dos dados**: assume `Snear/Smid/Sfar` com `nm-1` amostras
  (convenção SeReM). Há um `assert` que verifica `Nd == ntheta*(nm-1)`.
- **GPU**: se CUDA não estiver disponível, cai automaticamente para CPU.
- **Geradores aleatórios**: MATLAB e NumPy/PyTorch diferem; a comparação entre
  ambientes é estatística (faixas de convergência), não ponto a ponto.

## Referências

- Grana, D., Mukerji, T., Doyen, P. (2021). *Seismic Reservoir Modeling*.
  Wiley. (biblioteca SeReM/SeReMpy)
- Hansen, N., Niederberger, A. S. P., Guzzella, L., Koumoutsakos, P. (2009).
  *A Method for Handling Uncertainty in Evolutionary Optimization With an
  Application to Feedback Control of Combustion*. IEEE TEVC. (UH-CMA-ES)
