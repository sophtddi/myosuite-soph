import numpy as np
from myosuite.utils import gym

def evaluate_model(model, weighted_reward_keys=None, n_episodes=20, seed=999):
    """Évalue un modèle sur l'env d'imitation, avec des métriques neutres."""
    kwargs = {"seed": seed}
    if weighted_reward_keys is not None:
        kwargs["weighted_reward_keys"] = weighted_reward_keys

    env = gym.make('myoOSLRunTrackImitation-v0', **kwargs)
    distances, lengths, falls = [], [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        distances.append(env.unwrapped.obs_dict["model_root_pos"][1])
        lengths.append(steps)
        falls.append(env.unwrapped._get_fallen_condition())

    env.close()
    return {
        "distance_mean": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "ep_len_mean": float(np.mean(lengths)),
        "ep_len_std": float(np.std(lengths)),
        "fall_rate": float(np.mean(falls)),
    }