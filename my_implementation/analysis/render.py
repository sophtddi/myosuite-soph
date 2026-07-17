import os
import numpy as np
import imageio
import tqdm
from stable_baselines3 import PPO
from myosuite.utils import gym

def render_run_video(weighted_reward_keys, model_path, savepath, n_episodes=5, max_steps=1000):
    env = gym.make('myoOSLRunTrackImitation-v0', weighted_reward_keys=weighted_reward_keys)
    model = PPO.load(model_path, env=env, device="cpu")

    # transparence du groupe de géométries 1 — une fois suffit
    geom_1_indices = np.where(env.unwrapped.mj_model.geom_group == 1)
    env.unwrapped.mj_model.geom_rgba[geom_1_indices, 3] = 0

    frames, all_rewards = [], []
    for ep in tqdm.tqdm(range(n_episodes)):
        obs, info = env.reset()
        ep_rewards = []
        terminated, truncated, step_count = False, False, 0

        while not (terminated or truncated) and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            frames.append(env.unwrapped.mj_renderer.render_offscreen(width=400, height=400, camera_id=1))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rewards.append(reward)
            step_count += 1

        all_rewards.append(np.sum(ep_rewards))
        print(f"Episode {ep}: reward={np.sum(ep_rewards):.2f}, steps={step_count}")

    env.close()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    imageio.mimwrite(savepath, frames, fps=1.0 / env.unwrapped.dt)
    print(f"Average reward: {np.mean(all_rewards):.2f} over {len(all_rewards)} épisodes")
    return all_rewards