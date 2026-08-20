import json
import re
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from scipy.signal import butter, filtfilt
from mpl_toolkits.axes_grid1 import make_axes_locatable

UHCMAES_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(UHCMAES_DIR))

import uhcmaes_gpu


def carregar_mat(caminho_arquivo):
    """
    Carrega arquivos .mat binários (SciPy/MATLAB) e, caso seja um arquivo 
    de texto legado do Octave, faz o fallback para o parser customizado.
    """
    caminho_path = Path(caminho_arquivo)
    
    try:
        return sio.loadmat(caminho_path.as_posix())
    except Exception:
        dados = {}
        with open(caminho_path, 'r', encoding='latin-1', errors='ignore') as f:
            linhas = f.readlines()
            
        idx = 0
        while idx < len(linhas):
            linha = linhas[idx].strip()
            if linha.startswith("# name:"):
                nome_var = linha.split("# name:")[1].strip()
                idx += 1
                linha_tipo = linhas[idx].strip()
                tipo_var = linha_tipo.split("# type:")[1].strip()
                
                if tipo_var == "matrix":
                    idx += 1
                    rows = int(linhas[idx].strip().split("# rows:")[1])
                    idx += 1
                    cols = int(linhas[idx].strip().split("# columns:")[1])
                    idx += 1
                    matriz = []
                    for _ in range(rows):
                        valores = [float(x) for x in linhas[idx].strip().split()]
                        matriz.append(valores)
                        idx += 1
                    dados[nome_var] = np.array(matriz)
                    continue
                    
                elif tipo_var == "complex matrix":
                    idx += 1
                    rows = int(linhas[idx].strip().split("# rows:")[1])
                    idx += 1
                    cols = int(linhas[idx].strip().split("# columns:")[1])
                    idx += 1
                    matriz = []
                    for _ in range(rows):
                        linha_str = linhas[idx].strip()
                        valores = []
                        for token in linha_str.split():
                            token = token.replace('(', '').replace(')', '')
                            partes = token.split(',')
                            real = float(partes[0])
                            imag = float(partes[1]) if len(partes) > 1 else 0.0
                            valores.append(complex(real, imag))
                        matriz.append(valores)
                        idx += 1
                    dados[nome_var] = np.array(matriz)
                    continue
                    
                elif tipo_var == "scalar struct":
                    idx += 3  
                    continue
            idx += 1
        return dados


def plot_secao_2d(pasta_resultados_raiz, pasta_dados_originais, percentil_corte=2.0):
    """
    Varre as pastas de experimentos geradas em lote, extrai as propriedades físicas 
    e respostas sísmicas (usando SeismicPhysicsGPU) e plota os painéis comparativos.
    O parâmetro `percentil_corte` define a porcentagem de valores extremos a serem 
    ignorados na definição das escalas de cor (vmin, vmax).
    """
    dados_acumulados = {}
    vetor_tempo = None
    regex_coords = re.compile(r"IL(\d+)_XL(\d+)")
    
    pasta_resultados_path = Path(pasta_resultados_raiz)
    pasta_dados_path = Path(pasta_dados_originais)
    
    print("--- Iniciando Varredura de Resultados em Lote ---")
    
    for mat_file in pasta_resultados_path.rglob("run_data.mat"):
        mat_path_str = mat_file.as_posix()
        pasta_experimento = mat_file.parent
        
        json_file = pasta_experimento / "config_used.json"
        if not json_file.exists():
            json_file = pasta_experimento / "config.json"
        
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f_json:
                cfg = json.load(f_json)
            
            nome_input_mat = cfg.get("files", {}).get("input_filename", "")
            root_str = pasta_experimento.as_posix()
            
            match = regex_coords.search(nome_input_mat) or regex_coords.search(root_str)
            
            if not match:
                print(f"[PULADO] Nenhuma coordenada IL/XL encontrada em: {root_str}")
                continue
            
            inline_atual = int(match.group(1))
            xl_atual = int(match.group(2))
            
            resultado_mat = carregar_mat(mat_path_str)
            
            if "x_new" not in resultado_mat:
                print(f"[PULADO] Matriz 'x_new' não encontrada em: {mat_path_str}")
                continue
                
            x_new = resultado_mat["x_new"].flatten()
            
            if "fitness" in resultado_mat:
                fitness_history = resultado_mat["fitness"].flatten()
            elif "history" in resultado_mat and hasattr(resultado_mat["history"], "dtype"):
                fitness_history = resultado_mat["history"]["fitness"][0,0].flatten()
            else:
                fitness_history = np.array([0.0])
                
            final_fitness = np.abs(fitness_history[-1])
            
            arquivo_dado_real = f"data_segy_IL{inline_atual}_XL{xl_atual}.mat"
            caminho_dado_original = pasta_dados_path / arquivo_dado_real
            
            if not caminho_dado_original.exists():
                caminho_dado_original = pasta_dados_path / nome_input_mat
                
            if not caminho_dado_original.exists():
                print(f"[AVISO] Dado original não encontrado para XL {xl_atual} em: {caminho_dado_original.as_posix()}")
                continue
                
            dado_original = sio.loadmat(caminho_dado_original.as_posix())
            snear = dado_original["Snear"].flatten()
            vp_real_completo = dado_original["Vp"].flatten()
            vs_real_completo = dado_original["Vs"].flatten()
            rho_real_completo = dado_original["Rho"].flatten()
            time_seis_completo = dado_original["TimeSeis"].flatten()
            
            nm = len(vp_real_completo)
            limit_plot = nm - 1
            
            if vetor_tempo is None:
                vetor_tempo = time_seis_completo[0:limit_plot]
            
            vp_real = vp_real_completo[0:limit_plot]
            vs_real = vs_real_completo[0:limit_plot]
            rho_real = rho_real_completo[0:limit_plot]
            
            vp_sol_full = x_new[0 : nm]
            vs_sol_full = x_new[nm : 2*nm]
            rho_sol_full = x_new[2*nm : 3*nm]
            
            physics = uhcmaes_gpu.SeismicPhysicsGPU(
                cfg, 
                vp_real_completo, 
                vs_real_completo, 
                rho_real_completo, 
                time_seis_completo, 
                device="cpu", 
                dtype=torch.float64
            )
            
            vp_sol_n = physics.normalize(vp_sol_full, physics.p_norm["vp"])
            vs_sol_n = physics.normalize(vs_sol_full, physics.p_norm["vs"])
            rho_sol_n = physics.normalize(rho_sol_full, physics.p_norm["rho"])
            
            x_sol_n = torch.tensor(
                np.concatenate([vp_sol_n, vs_sol_n, rho_sol_n]), 
                dtype=torch.float64, 
                device="cpu"
            ).unsqueeze(1)
            
            y_pred = physics.forward_batch(x_sol_n).detach().cpu().squeeze().numpy()
            
            seismic_real = snear[0:limit_plot]
            seismic_sol = y_pred[0:limit_plot]  
            
            vp_sol = vp_sol_full[0:limit_plot]
            vs_sol = vs_sol_full[0:limit_plot]
            rho_sol = rho_sol_full[0:limit_plot]
            
            nfilt = int(cfg["physics"]["prior_filter_order"])
            cutofffr = float(cfg["physics"]["prior_cutoff_freq"])
            b, a = butter(nfilt, cutofffr)
            vp_prior = filtfilt(b, a, vp_real)
            vs_prior = filtfilt(b, a, vs_real)
            rho_prior = filtfilt(b, a, rho_real)
            
            dados_acumulados[xl_atual] = {
                "seismic_real": seismic_real, "seismic_sol": seismic_sol,
                "vp_real": vp_real, "vp_prior": vp_prior, "vp_sol": vp_sol,
                "vs_real": vs_real, "vs_prior": vs_prior, "vs_sol": vs_sol,
                "rho_real": rho_real, "rho_prior": rho_prior, "rho_sol": rho_sol,
                "final_fitness": final_fitness
            }
            print(f"[CARREGADO] XL {xl_atual} com sucesso ({root_str})")

    if not dados_acumulados:
        print("Nenhum dado válido de experimento pôde ser processado.")
        return

    print(f"\nTotal de Crosslines carregadas para a seção 2D: {len(dados_acumulados)}")
    crosslines_ordenadas = sorted(dados_acumulados.keys())
    n_xl = len(crosslines_ordenadas)
    n_z = len(vetor_tempo)
    
    secoes_2d = {nome: np.zeros((n_z, n_xl)) for nome in [
        "seismic_real", "seismic_sol",
        "vp_real", "vp_prior", "vp_sol",
        "vs_real", "vs_prior", "vs_sol",
        "rho_real", "rho_prior", "rho_sol"
    ]}
    
    vetor_diagnostico_fit = np.zeros(n_xl)

    for col_idx, xl in enumerate(crosslines_ordenadas):
        vetor_diagnostico_fit[col_idx] = dados_acumulados[xl]["final_fitness"]
        for chave in secoes_2d.keys():
            secoes_2d[chave][:, col_idx] = dados_acumulados[xl][chave]

    if n_xl == 1:
        xl_val = crosslines_ordenadas[0]
        extent = [xl_val - 0.5, xl_val + 0.5, vetor_tempo[-1], vetor_tempo[0]]
    else:
        extent = [crosslines_ordenadas[0], crosslines_ordenadas[-1], vetor_tempo[-1], vetor_tempo[0]]

    # Plot de Convergência Regional
    plt.figure(figsize=(10, 3))
    plt.plot(crosslines_ordenadas, vetor_diagnostico_fit, "o-", color="purple", linewidth=2, label="Misfit Final")
    plt.yscale("log")
    plt.title(f"Diagnóstico Regional de Inversão - Inline {inline_atual}")
    plt.xlabel("Crossline (Perfil Lateral)")
    plt.ylabel("Função Objetivo")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()

    # Configuração dos painéis 2x3
    propriedades = [
        ("Vp (Velocidade P - km/s)", "vp_real", "vp_sol", "jet"),
        ("Vs (Velocidade S - km/s)", "vs_real", "vs_sol", "plasma"),
        ("Rho (Densidade - g/cc)", "rho_real", "rho_sol", "viridis")
    ]

    for label_nome, ref_real, ref_sol, mapa_cor in propriedades:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
        fig.suptitle(f"Perfil Comparativo - {label_nome}", fontsize=14, fontweight="bold")
        
        # =========================================================
        # CÁLCULO DE LIMITES ROBUSTOS (REMOVENDO EXTREMOS VIA PERCENTIL)
        # =========================================================
        # Para sinais simétricos ao redor de zero (sísmica) usamos os absolutos
        lim_seis_real = np.percentile(np.abs(secoes_2d["seismic_real"]), 100 - percentil_corte)
        lim_seis_sol = np.percentile(np.abs(secoes_2d["seismic_sol"]), 100 - percentil_corte)
        lim_seis = max(lim_seis_real, lim_seis_sol)
        if lim_seis == 0: lim_seis = 1e-5
        
        # Para propriedades geológicas, fazemos o corte superior e inferior
        vmin_prop = np.percentile(secoes_2d[ref_real], percentil_corte)
        vmax_prop = np.percentile(secoes_2d[ref_real], 100 - percentil_corte)

        # =========================================================
        # LINHA 1: SÍSMICA
        # =========================================================
        
        # 1. Sísmica Real (Ground Truth)
        im_s_gt = axes[0, 0].imshow(secoes_2d["seismic_real"], cmap="seismic", aspect="auto", extent=extent, vmin=-lim_seis, vmax=lim_seis)
        axes[0, 0].set_title("Sísmica Ground Truth", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Tempo de Trânsito (ms)")
        
        # 2. Sísmica Encontrada
        axes[0, 1].imshow(secoes_2d["seismic_sol"], cmap="seismic", aspect="auto", extent=extent, vmin=-lim_seis, vmax=lim_seis)
        axes[0, 1].set_title("Sísmica Encontrada", fontsize=11, fontweight="bold")
        
        # 3. Resíduo Sísmico
        res_seis = secoes_2d["seismic_sol"] - secoes_2d["seismic_real"]
        lim_res_seis = np.percentile(np.abs(res_seis), 100 - percentil_corte)
        if lim_res_seis == 0: lim_res_seis = 1e-5
        im_s_res = axes[0, 2].imshow(res_seis, cmap="seismic", aspect="auto", extent=extent, vmin=-lim_res_seis, vmax=lim_res_seis)
        axes[0, 2].set_title("Resíduo da Sísmica", fontsize=11, fontweight="bold", color="darkred")
        
        # Colorbars da Linha 1
        div_s_gt = make_axes_locatable(axes[0, 1])
        cax_s_gt = div_s_gt.append_axes("right", size="5%", pad=0.08)
        cbar_s_gt = fig.colorbar(im_s_gt, cax=cax_s_gt)
        cbar_s_gt.set_label("Amplitude Sísmica", fontsize=9)
        cbar_s_gt.ax.tick_params(labelsize=8)

        div_s_res = make_axes_locatable(axes[0, 2])
        cax_s_res = div_s_res.append_axes("right", size="5%", pad=0.08)
        cbar_s_res = fig.colorbar(im_s_res, cax=cax_s_res)
        cbar_s_res.set_label("Erro Absoluto (Δ)", fontsize=9)
        cbar_s_res.ax.tick_params(labelsize=8)


        # =========================================================
        # LINHA 2: PROPRIEDADES FÍSICAS
        # =========================================================
        
        # 1. Propriedade Ground Truth
        im_p_gt = axes[1, 0].imshow(secoes_2d[ref_real], cmap=mapa_cor, aspect="auto", extent=extent, vmin=vmin_prop, vmax=vmax_prop)
        axes[1, 0].set_title("Propriedade Ground Truth", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Crossline")
        axes[1, 0].set_ylabel("Tempo de Trânsito (ms)")
        
        # 2. Propriedade Encontrada
        axes[1, 1].imshow(secoes_2d[ref_sol], cmap=mapa_cor, aspect="auto", extent=extent, vmin=vmin_prop, vmax=vmax_prop)
        axes[1, 1].set_title("Propriedade Encontrada", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Crossline")
        
        # 3. Resíduo da Propriedade
        res_prop = secoes_2d[ref_sol] - secoes_2d[ref_real]
        lim_res_prop = np.percentile(np.abs(res_prop), 100 - percentil_corte)
        if lim_res_prop == 0: lim_res_prop = 1e-5
        im_p_res = axes[1, 2].imshow(res_prop, cmap="seismic", aspect="auto", extent=extent, vmin=-lim_res_prop, vmax=lim_res_prop)
        axes[1, 2].set_title("Resíduo da Propriedade", fontsize=11, fontweight="bold", color="darkred")
        axes[1, 2].set_xlabel("Crossline")
        
        # Colorbars da Linha 2
        div_p_gt = make_axes_locatable(axes[1, 1])
        cax_p_gt = div_p_gt.append_axes("right", size="5%", pad=0.08)
        cbar_p_gt = fig.colorbar(im_p_gt, cax=cax_p_gt)
        cbar_p_gt.set_label(label_nome, fontsize=9)
        cbar_p_gt.ax.tick_params(labelsize=8)

        div_p_res = make_axes_locatable(axes[1, 2])
        cax_p_res = div_p_res.append_axes("right", size="5%", pad=0.08)
        cbar_p_res = fig.colorbar(im_p_res, cax=cax_p_res)
        cbar_p_res.set_label("Erro Absoluto (Δ)", fontsize=9)
        cbar_p_res.ax.tick_params(labelsize=8)
        
        for ax in axes.flat:
            ax.invert_yaxis()

        plt.tight_layout()

    plt.show()


ROOT_DIR = Path("C:/Users/diogo/Projetos/uhcmaes-gpu")
RESULTS_DIR = (ROOT_DIR / "Results_UHCMAES_py/unisim_crop_2d").as_posix()
DATA_DIR = (ROOT_DIR / "examples/UNISIM_IL118").as_posix()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plota secção 2d")
    parser.add_argument("--results_folder", default=RESULTS_DIR, help="Diretório com resultados armazenados")
    parser.add_argument("--data_folder", default=DATA_DIR, help="Diretório com dados armazenados")
    parser.add_argument("--corte_percentil", type=float, default=2.0, help="Porcentagem de outliers cortados na paleta")
    args = parser.parse_args()

    pasta_resultados = Path(args.results_folder).as_posix()
    pasta_dados_originais = Path(args.data_folder).as_posix()

    plot_secao_2d(pasta_resultados, pasta_dados_originais, percentil_corte=args.corte_percentil)