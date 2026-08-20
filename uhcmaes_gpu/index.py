import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from uhcmaes_gpu import run_uhcmaes_gpu

ROOT_DIR = Path("C:/Users/diogo/Projetos/uhcmaes-gpu")
DATA_DIR = (ROOT_DIR / "examples/UNISIM_IL118").as_posix()
RESULTS_DIR = (ROOT_DIR / "Results_UHCMAES_py/unisim_crop_2d").as_posix()
CONFIG_PATH = (ROOT_DIR / "parameters/config_unisim_1559.json").as_posix()

def run_single_file(
        mat_path, 
        config_path=CONFIG_PATH,
        seed=42
    ):
    try:
        mat_path_obj = Path(mat_path)
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "files" not in cfg:
            cfg["files"] = {}
        cfg["files"]["input_filename"] = mat_path_obj.name

        out = run_uhcmaes_gpu(
            cfg, 
            mat_path_obj.as_posix(), 
            results_folder=RESULTS_DIR, 
            device="cuda", 
            seed=seed
        )

        return {
            "file": str(mat_path),
            "rel_error": out["best"]["rel_error"],
            "status": "success"
        }
    except Exception as e:
        return {
            "file": str(mat_path),
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    data_dir = Path(DATA_DIR).resolve()
    mat_files = list(data_dir.glob("*.mat"))

    MAX_WORKERS = 5

    print(f"Iniciando processamento em paralelo de {len(mat_files)} arquivos...")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(run_single_file, filepath): filepath 
            for filepath in mat_files
        }

        for future in as_completed(future_to_file):
            res = future.result()
            if res["status"] == "success":
                print(f"[CONCLUÍDO] {res['file']} | Melhor rel_error: {res['rel_error']:.6f}")
                results.append(res)
            else:
                print(f"[ERRO] {res['file']} | Detalhe: {res['error']}")