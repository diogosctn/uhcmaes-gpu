# Comparação cruzada MATLAB vs Python — UH-CMA-ES

Este diretório roda o **mesmo `config.json`** e a **mesma seed** nos dois ambientes
e compara os resultados, validando se a versão Python (`uhcmaes_gpu.py`) resolve o
mesmo problema da versão MATLAB (`UHCMAES.m`).

## Arquivos

| Arquivo | Função |
|---|---|
| `config.json` | Configuração canônica (mesmo schema do MATLAB) + bloco `compare` com a `seed` |
| `run_matlab.m` | Runner MATLAB/Octave (réplica fiel do `UHCMAES.m`, instrumentado) → `out_matlab.mat` |
| `run_python.py` | Runner Python (usa `uhcmaes_gpu.py`) → `out_python.mat` |
| `compare_runs.py` | Lê os dois `.mat` e gera relatório + 3 figuras |
| `SeReM/` | `RickerWavelet.m`, `WaveletMatrix.m`, `DifferentialMatrix.m` (oficiais) |
| `matlab/` | `UHCMAES.m`, `SeismicModel.m`, `AkiRichardsCoefficientsMatrix.m` |
| `data_segy_IL118_XL229.mat` | Dado real (Vp, Vs, Rho, TimeSeis, Snear/Smid/Sfar) |

## Como rodar

```bash
# 1) MATLAB (ou Octave, com pacotes signal/statistics/communications/io)
octave --quiet run_matlab.m        # ou, no MATLAB: run_matlab

# 2) Python
python3 run_python.py --device cuda --dtype float64

# 3) Comparação
python3 compare_runs.py
```

Saídas: `compare_report.txt`, `compare_fig1_prior.png`,
`compare_fig2_convergence.png`, `compare_fig3_solution.png`.

## O que é comparado

1. **Sísmico do prior** — isola o modelo direto (forward). Deve ser ~idêntico.
2. **Perfis do prior** (Vp/Vs/Rho após Butterworth) — valida o pré-processamento.
3. **Sísmico do melhor indivíduo da 1ª geração** — apenas escala/distribuição
   (os geradores aleatórios do MATLAB e do NumPy/PyTorch são diferentes).
4. **Trajetórias de convergência** (fitness, sigma, rel_error, correlation) —
   devem convergir para faixas semelhantes (equivalência estatística).
5. **Solução final** (perfis invertidos) — deve ter correlação alta.

## Resultado obtido neste dado

| Métrica | Resultado | Interpretação |
|---|---|---|
| Sísmico do prior | `rel_err = 4.7e-3`, `corr = 0.99999` | Forward equivalente (diferença residual vem do `filtfilt` Octave vs SciPy) |
| Perfis do prior | `rel_err ≈ 3–6e-4`, `corr > 0.9999` | Pré-processamento equivalente |
| Melhor rel_error | MATLAB 0.9876 / Python 0.9888 | Mesma faixa de convergência |
| Solução final | `corr = 0.97–0.98` (Vp, Vs, Rho) | Perfis invertidos muito semelhantes |

## Notas importantes

- **Correção no `AkiRichardsCoefficientsMatrix.m`**: a linha original
  `avgVs.^2. / avgVp.^2` (com espaço antes de `/`) é interpretada pelo **Octave**
  como **divisão matricial** (`/`), gerando erro de dimensão. Foi corrigido para
  `avgVs.^2 ./ avgVp.^2` (divisão elemento a elemento), que é o comportamento
  pretendido e o que o MATLAB executa. **Atenção**: se você rodar no MATLAB
  original, essa correção também é recomendada (o MATLAB tolera, mas o Octave não).
- **Geradores aleatórios**: MATLAB e NumPy/PyTorch usam algoritmos diferentes.
  Assim, mesmo com a mesma seed, os indivíduos amostrados **não são os mesmos** —
  a comparação é **estatística** (faixas de convergência), não ponto a ponto.
- **`filtfilt`**: Octave e SciPy diferem levemente no padding de borda, o que
  propaga uma diferença de ~5e-4 no prior e ~5e-3 no sísmico do prior. É
  numericamente desprezível e não afeta a convergência.
