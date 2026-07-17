import os
import sys

# Ajoute le dossier racine du projet et le sous-dossier myosuite au PATH de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "myosuite"))


from myosuite.my_implementation.training.trainer import train_model_imitation

if __name__ == "__main__":
    model = train_model_imitation(
        run_name="testALLLEZ",
        weighted_reward_keys={"sparse": 0, "solved": 10, "qpos_imitation": 30, "forward_bounded": 10, "alive_bonus": 15,
        "fall_penalty": -50,},
        total_timesteps=5_000_000,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        seed=42,
        n_envs=8,
        # resume_from="imitation_init0_sparse1_solved10_qpos40",
    )

