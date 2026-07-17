import os
import pandas as pd
from stable_baselines3 import PPO

from my_implementation.training.checkpoints import find_latest_checkpoint
from my_implementation.analysis.evaluate import evaluate_model

def summary_table(run_names, checkpoints_dir, savepath, n_episodes=20):
    rows = []
    for run_name in run_names:
        ckpt_path = find_latest_checkpoint(run_name, checkpoints_dir)
        if ckpt_path is None:
            print(f"[!] Aucun checkpoint trouvé pour {run_name}")
            continue

        print(f"Évaluation de {run_name} (checkpoint: {os.path.basename(ckpt_path)})...")
        model = PPO.load(ckpt_path, device="cpu")
        metrics = evaluate_model(model, n_episodes=n_episodes)
        metrics["run_name"] = run_name
        metrics["checkpoint"] = os.path.basename(ckpt_path)
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("run_name")
    df.to_csv(savepath)
    print(df)
    return df