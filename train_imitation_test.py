import os
import glob
import numpy as np
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import osl_imitation_env


class ResumableEvalCallback(EvalCallback):
    """EvalCallback qui recharge l'historique existant plutôt que de l'écraser."""
    def _init_callback(self) -> None:
        super()._init_callback()
        if self.log_path is not None:
            npz_file = self.log_path + ".npz"
            if os.path.exists(npz_file):
                data = np.load(npz_file)
                self.evaluations_timesteps = list(data["timesteps"])
                self.evaluations_results = list(data["results"])
                self.evaluations_length = list(data["ep_lengths"])
                print(f"[ResumableEvalCallback] {len(self.evaluations_timesteps)} évaluations précédentes rechargées depuis {npz_file}")


def find_latest_checkpoint(run_name, checkpoints_dir="./checkpoints"):
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


def train_model_imitation(run_name, weighted_reward_keys, total_timesteps, seed,
                           reset_type="random", net_arch=None, n_envs=8,
                           resume_from=None):

    def make_env():
        env = gym.make('myoOslImitation-v0', weighted_reward_keys=weighted_reward_keys, reset_type=reset_type)
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 1000)

    eval_callback = ResumableEvalCallback(   # <-- changé ici
        eval_env,
        best_model_save_path=f"./checkpoints/{run_name}/",
        log_path=f"./eval_logs/{run_name}/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path=f"./checkpoints/{run_name}/",
        name_prefix=run_name,
    )

    reset_num_timesteps = True
    if resume_from is not None:
        if resume_from.endswith(".zip") or os.path.exists(resume_from):
            checkpoint_path = resume_from
        else:
            checkpoint_path = find_latest_checkpoint(resume_from)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Aucun checkpoint trouvé pour resume_from='{resume_from}'")
        print(f"[Reprise] Chargement depuis : {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=train_env)
        reset_num_timesteps = False
    else:
        print(f"[Nouveau run] Création d'un modèle from scratch pour '{run_name}'")
        policy_kwargs = dict(net_arch=net_arch) if net_arch is not None else {}
        model = PPO("MlpPolicy", train_env, n_steps=2048, batch_size=256,
                     policy_kwargs=policy_kwargs, verbose=1, seed=seed,
                     tensorboard_log="./tb_logs/")

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
    model = train_model_imitation(
        run_name="imitation_initR_sparse0_solved10_qpos40_forward5",
        weighted_reward_keys={"sparse": 0, "solved": 10, "qpos_imitation": 40, "forward_bounded": 5},
        total_timesteps=8000,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        seed=42,
        n_envs=8,
        # resume_from="imitation_init0_sparse1_solved10_qpos40",
    )