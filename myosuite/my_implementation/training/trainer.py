import os
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from myosuite.my_implementation.training.callbacks import ResumableEvalCallback
from myosuite.my_implementation.training.checkpoints import find_latest_checkpoint
from myosuite.my_implementation.utils.paths import get_results_root, get_local_logs_root


def train_model_imitation(run_name, weighted_reward_keys, total_timesteps, seed,
                           reset_type="random", net_arch=None, n_envs=8,
                           resume_from=None):

    results_root = get_results_root()
    checkpoints_dir = results_root / "checkpoints"
    eval_logs_dir = results_root / "eval_logs"
    tb_logs_dir = get_local_logs_root()

    def make_env():
        env = gym.make('myoOSLRunTrackImitation-v0', weighted_reward_keys=weighted_reward_keys, reset_type=reset_type)
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 1000)

    eval_callback = ResumableEvalCallback(
        eval_env,
        best_model_save_path=str(checkpoints_dir / run_name),
        log_path=str(eval_logs_dir / run_name),
        eval_freq=500_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path=str(checkpoints_dir / run_name),
        name_prefix=run_name,
    )

    reset_num_timesteps = True
    if resume_from is not None:
        checkpoint_path = resume_from if (resume_from.endswith(".zip") or os.path.exists(resume_from)) \
            else find_latest_checkpoint(resume_from, checkpoints_dir)
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
                     tensorboard_log=str(tb_logs_dir), device="cpu")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=run_name,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(str(checkpoints_dir / f"{run_name}_final"))

    train_env.close()
    eval_env.close()
    return model


