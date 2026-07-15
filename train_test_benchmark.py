import os
import numpy as np
import json
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import glob

def find_latest_checkpoint(run_name, checkpoints_dir="./checkpoints"):
    """Trouve le checkpoint le plus avancé (le plus de steps) pour un run donné."""
    pattern = f"{checkpoints_dir}/{run_name}/{run_name}_*_steps.zip"
    candidates = glob.glob(pattern)
    if not candidates:
        for fallback in [f"{checkpoints_dir}/{run_name}_final.zip",
                          f"{checkpoints_dir}/{run_name}/best_model.zip"]:
            if os.path.exists(fallback):
                return fallback
        return None
    candidates.sort(key=lambda p: int(p.split("_")[-2]))
    return candidates[-1]


def train_model(run_name, weighted_reward_keys, total_timesteps, seed,
                 reset_type="random", net_arch=None, n_envs=8,
                 resume_from=None):
    """
    resume_from : soit
        - None -> entraîne un nouveau modèle from scratch
        - un chemin de fichier .zip précis -> reprend depuis ce checkpoint
        - un run_name existant -> reprend depuis son dernier checkpoint automatiquement
    """

    def make_env():
        env = gym.make(
            'myoChallengeOslRunFixed-v0',
            weighted_reward_keys=weighted_reward_keys,
            reset_type=reset_type,
        )
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 1000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./checkpoints/{run_name}/",
        log_path=f"./eval_logs/{run_name}/",
        eval_freq=500_000 // n_envs,
        n_eval_episodes=20,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path=f"./checkpoints/{run_name}/",
        name_prefix=run_name,
    )

    # --- Reprise ou nouveau modèle ---
    reset_num_timesteps = True
    if resume_from is not None:
        # Si c'est un chemin .zip direct, on l'utilise tel quel.
        # Sinon on suppose que c'est un run_name, et on va chercher son dernier checkpoint.
        if resume_from.endswith(".zip") or os.path.exists(resume_from):
            checkpoint_path = resume_from
        else:
            checkpoint_path = find_latest_checkpoint(resume_from)

        if checkpoint_path is None:
            raise FileNotFoundError(f"Aucun checkpoint trouvé pour resume_from='{resume_from}'")

        print(f"[Reprise] Chargement depuis : {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=train_env)
        reset_num_timesteps = False  # continue le compteur de pas, ne repart pas de 0
    else:
        print(f"[Nouveau run] Création d'un modèle from scratch pour '{run_name}'")
        policy_kwargs = dict(net_arch=net_arch) if net_arch is not None else {}
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=2048,
            batch_size=256,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log="./tb_logs/",
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=run_name,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(f"./checkpoints/{run_name}_final")
    train_env.close()
    eval_env.close()

    return model

if __name__ == "__main__":

    model = train_model(
    run_name="baseline_sparse1_solved10_random_net256",  # <-- SANS suffixe "_1M"/"_10M", même nom à chaque reprise
    weighted_reward_keys={"sparse": 1, "solved": 10},
    total_timesteps=10_000_000,  # pas restants
    seed=42,
    reset_type="random",
    resume_from="baseline_sparse1_solved10_random_net256",  # pointe vers l'ancien nom, une seule fois, pour charger le poids initial
)
    
    # model = train_model(
    #     run_name="baseline_sparse1_solved10_random_net256_1M",
    #     weighted_reward_keys={"sparse": 1, "solved": 10},
    #     total_timesteps=1_000_000,
    #     seed=42,
    #     reset_type="random",                      # pose fixe au reset, plus simple pour apprendre la base
    #     net_arch=dict(pi=[256, 256], vf=[256, 256]),  # réseau plus grand
    #     n_envs=8,
    # )