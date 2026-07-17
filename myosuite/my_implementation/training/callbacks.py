import os
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


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