import os
import numpy as np
import matplotlib.pyplot as plt

def plot_training_comparison(run_names, eval_logs_dir, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for run_name in run_names:
        npz_path = os.path.join(str(eval_logs_dir), run_name, "evaluations.npz")
        if not os.path.exists(npz_path):
            print(f"[!] Pas de fichier trouvé pour {run_name} à {npz_path}")
            continue

        data = np.load(npz_path)
        timesteps, results, ep_lengths = data["timesteps"], data["results"], data["ep_lengths"]
        reward_mean, reward_std = results.mean(axis=1), results.std(axis=1)
        len_mean = ep_lengths.mean(axis=1)

        axes[0].plot(timesteps, reward_mean, label=run_name)
        axes[0].fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
        axes[1].plot(timesteps, len_mean, label=run_name)
        axes[2].plot(timesteps, reward_std, label=run_name)

    axes[0].set_title("Reward moyen (éval)"); axes[0].set_xlabel("Pas d'entraînement")
    axes[1].set_title("Durée moyenne des épisodes"); axes[1].set_xlabel("Pas d'entraînement")
    axes[2].set_title("Écart-type du reward"); axes[2].set_xlabel("Pas d'entraînement")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    return fig