from my_implementation.training.trainer import train_model_imitation

if __name__ == "__main__":
    model = train_model_imitation(
        run_name="test",
        weighted_reward_keys={"sparse": 0, "solved": 10, "qpos_imitation": 30, "forward_bounded": 10, "alive_bonus": 15,
        "fall_penalty": -50,},
        total_timesteps=1_000_000,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        seed=42,
        n_envs=8,
        # resume_from="imitation_init0_sparse1_solved10_qpos40",
    )

