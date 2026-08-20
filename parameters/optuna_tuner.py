import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import optuna

UHCMAES_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(UHCMAES_DIR))

import uhcmaes_gpu


ROOT_DIR = "C:/Users/diogo/Projetos/uhcmaes-gpu"
PARAM_DIR = ROOT_DIR + "/parameters"
CONFIG_PATH = PARAM_DIR + "/config_unisim.json"
RESULTS_BASE =  PARAM_DIR + "/Results_UHCMAES_unisim_parametrization_tuner/"

def objective(trial):
    #wavelet_freq = trial.suggest_int("wavelet_freq", 20, 60, step=5)
    #wavelet_ntw = trial.suggest_categorical("wavelet_ntw", [32, 64, 128])
    #prior_filter_order = trial.suggest_int("prior_filter_order", 1, 5)
    #prior_cutoff_freq = trial.suggest_float("prior_cutoff_freq", 0.01, 0.1)
    #correlation_length_factor = trial.suggest_float("correlation_length_factor", 1.0, 10.0)
    normalization_method = trial.suggest_categorical("normalization_method", ["standard", "linear"])

    sigma_initial = trial.suggest_float("sigma_initial", 0.01, 5.0)
    stop_tol_diversity = trial.suggest_float("stop_tol_diversity", 0.01, 0.5)
    #gen_method = trial.suggest_categorical("gen_method", ["cmaes", "mvnrnd"])
    reg_type = trial.suggest_categorical("reg_type", ["const", "5_exp1", "5_exp001", "sigma_exp0001"])
    
    #stop_method = trial.suggest_categorical("stop_method", ["relative_error", "correlation", "chi_squared", "stagnation", "diversity"])
    
    # Condicionamento dos thresholds de parada dependendo da métrica escolhida
    #if stop_method == "relative_error":
    #    stop_threshold = trial.suggest_float("stop_threshold_rel_error", 0.1, 0.6)
    #elif stop_method == "correlation":
    #    stop_threshold = trial.suggest_float("stop_threshold_corr", 0.7, 0.99)
    #elif stop_method == "chi_squared":
    #    stop_threshold = trial.suggest_float("stop_threshold_chi", 0.1, 2.0)
    #elif stop_method == "stagnation":
    #    stop_threshold = trial.suggest_float("stop_threshold_stag", 1e-6, 1e-3, log=True)
    #else:  # diversity
    #    stop_threshold = trial.suggest_float("stop_threshold_div", 0.01, 0.2)
        
    #stagnation_window = trial.suggest_int("stagnation_window", 50, 200, step=25)

    # -- Parâmetros Uncertainty Handling (UH) --
    noise_level = trial.suggest_float("noise_level", 0.01, 0.5)
    r_lambda = trial.suggest_float("r_lambda", 0.1, 0.5)
    theta_uh = trial.suggest_float("theta_uh", 0.05, 0.4)
    cs_uh = trial.suggest_float("cs_uh", 0.1, 0.5)
    alpha_t = trial.suggest_float("alpha_t", 1.1, 2.0)
    alpha_sigma = trial.suggest_float("alpha_sigma", 0.1, 0.9)
    #t_eval_initial = trial.suggest_int("t_eval_initial", 5, 50)
    
    # Condicionamento para garantir t_max > t_min e coerência com eval inicial
    #t_min = trial.suggest_int("t_min", 1, 10)
    #t_max = trial.suggest_int("t_max", max(t_min + 10, t_eval_initial), 100)

    # Carrega o json padrão apenas para manter as chaves que não mudam (files, nvars, angles)
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        # Fallback dictionary if file not found locally
        print("\n=== ERRO NA OTIMIZAÇÃO MASSIVA EM GPU ===")
                
    # Atualiza a estrutura com os novos parâmetros gerados
    #cfg["physics"]["wavelet_freq"] = wavelet_freq
    #cfg["physics"]["wavelet_ntw"] = wavelet_ntw
    #cfg["physics"]["prior_filter_order"] = prior_filter_order
    #cfg["physics"]["prior_cutoff_freq"] = prior_cutoff_freq
    #cfg["physics"]["correlation_length_factor"] = correlation_length_factor
    cfg["physics"]["normalization"]["method"] = normalization_method
    
    cfg["cmaes"]["sigma_initial"] = sigma_initial
    cfg["cmaes"]["stop_tol_diversity"] = stop_tol_diversity
    #cfg["cmaes"]["gen_method"] = gen_method
    cfg["cmaes"]["reg_type"] = reg_type
    
    #cfg["cmaes"]["stop_criteria"]["method"] = stop_method
    #cfg["cmaes"]["stop_criteria"]["threshold"] = stop_threshold
    #cfg["cmaes"]["stop_criteria"]["stagnation_window"] = stagnation_window
    
    cfg["uh"]["noise_level"] = noise_level
    cfg["uh"]["r_lambda"] = r_lambda
    cfg["uh"]["theta_uh"] = theta_uh
    cfg["uh"]["cs_uh"] = cs_uh
    cfg["uh"]["alpha_t"] = alpha_t
    cfg["uh"]["alpha_sigma"] = alpha_sigma
    #cfg["uh"]["t_eval_initial"] = t_eval_initial
    #cfg["uh"]["t_min"] = t_min
    #cfg["uh"]["t_max"] = t_max

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
        
    # 3. Execução do Python dentro da pasta Codes/UHCMAES
    try:
        uhcmaes_gpu.run_uhcmaes_gpu(
            cfg, 
            os.path.join(cfg["files"]["data_folder"], cfg["files"]["input_filename"]), 
            RESULTS_BASE
        )
    except subprocess.CalledProcessError:
        return float("inf")

    # 4. Leitura do log na pasta de resultados do Octave
    if not Path(RESULTS_BASE).resolve().exists():
        return float("inf")

    runs = [d for d in Path(RESULTS_BASE).resolve().iterdir() if d.is_dir()]
    if not runs:
        return float("inf")

    latest_run = max(runs, key=os.path.getmtime)
    csv_log = latest_run / "log_execucao.csv"

    try:
        df = pd.read_csv(csv_log)
        return df["RelError"].min()
    except Exception:
        return float("inf")

if __name__ == "__main__":
    print("Iniciando Otimização Completa com Optuna em GPU...")
    
    # Cria o banco de dados SQLite para salvar o progresso. 
    # Isso permite pausar, continuar e usar optuna-dashboard depois.
    study_name = "unisim"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
    )
    
    # Executa a busca massiva
    study.optimize(objective, n_trials=10000)

    print("\n=== OTIMIZAÇÃO MASSIVA EM GPU CONCLUÍDA ===")
    print(f"Melhor Erro Relativo: {study.best_value:.4f}")
    print("Melhores Parâmetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    print("\nPara visualizar os resultados graficamente, execute no terminal:")
    print(f"optuna-dashboard sqlite:///{study_name}.db")