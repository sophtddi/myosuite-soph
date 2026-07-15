import numpy as np
from myosuite.envs.myo.myochallenge.run_track_v0 import RunTrack
from gymnasium.envs.registration import register


class RunTrackImitation(RunTrack):

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "sparse": 0,           # <-- on neutralise l'ancien reward non borné
        "solved": 10,
        "qpos_imitation": 40,
        "forward_bounded": 5, # <-- nouveau reward de vitesse, borné
    }

    def _setup(self, **kwargs):
        self._imitation_index = 0
        self._target_velocity = 1.2

        super()._setup(**kwargs)

        # pelvis_vel_X dans le CSV = vitesse d'avancée (pas pelvis_vel_Y, qui est le mouvement latéral)
        self._target_velocity = np.mean(
            self.INIT_DATA[:, self.gait_cycle_headers['pelvis_vel_X']]
        )
        print(f"[RunTrackImitation] Vitesse cible recalculée depuis la référence : {self._target_velocity:.4f} m/s")

    def reset(self, **kwargs):
        self._imitation_index = self.np_random.integers(0, int(self.INIT_DATA.shape[0] * 0.8))
        obs = super().reset(**kwargs)
        self._set_pose_from_reference(self._imitation_index)
        return obs

    def _set_pose_from_reference(self, ref_idx):
        for jnt in self.biological_jnt:
            if jnt in self.gait_cycle_headers:
                ref_qpos = self.INIT_DATA[ref_idx, self.gait_cycle_headers[jnt]]
                self.mj_data.joint(jnt).qpos[0] = ref_qpos
        import mujoco
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def step(self, a, **kwargs):
        self._imitation_index += 1
        if self._imitation_index >= self.INIT_DATA.shape[0]:
            self._imitation_index = 0
        return super().step(a, **kwargs)

    def _get_qpos_diff_array(self):
        diffs = []
        for jnt in self.biological_jnt:
            if jnt in self.gait_cycle_headers:
                ref_val = self.INIT_DATA[self._imitation_index, self.gait_cycle_headers[jnt]]
                cur_val = self.mj_data.joint(jnt).qpos[0]
                diffs.append(cur_val - ref_val)
        return np.array(diffs)

    def _get_forward_bounded_reward(self):
        """
        Reward de vitesse borné entre 0 et dt, calé sur une vitesse cible fixe,
        au lieu du 'sparse' original (-vitesse, non borné).
        """
        current_vel = self.obs_dict["model_root_vel"].squeeze()[1]  # même signal que get_score()
        speed_error = (-current_vel) - self._target_velocity  # -current_vel car le parcours va vers -y
        return self.dt * np.exp(-5 * np.square(speed_error))

    def get_reward_dict(self, obs_dict):
        q_diff = self._get_qpos_diff_array()
        qpos_imitation_value = self.dt * np.mean(np.exp(-8 * np.square(q_diff)))
        forward_bounded_value = self._get_forward_bounded_reward()

        # Retire temporairement les clés custom pour ne pas faire planter le calcul dense du parent
        weights_to_restore = {}
        for key in ["qpos_imitation", "forward_bounded"]:
            w = self.rwd_keys_wt.pop(key, None)
            if w is not None:
                weights_to_restore[key] = w

        rwd_dict = super().get_reward_dict(obs_dict)  # calcule dense avec sparse=0, solved

        for key, w in weights_to_restore.items():
            self.rwd_keys_wt[key] = w

        rwd_dict["qpos_imitation"] = qpos_imitation_value
        rwd_dict["forward_bounded"] = forward_bounded_value
        rwd_dict["dense"] = (
            rwd_dict["dense"]
            + weights_to_restore.get("qpos_imitation", 0) * qpos_imitation_value
            + weights_to_restore.get("forward_bounded", 0) * forward_bounded_value
        )
        return rwd_dict


register(
    id='myoOslImitation-v0',
    entry_point='osl_imitation_env:RunTrackImitation',
    kwargs={
        "model_path": r"c:\Users\ST000082\Documents\Codes\myosuite\myosuite\envs\myo\myochallenge\..\assets\leg\myoosl_runtrack.xml",
        "init_pose_path": r"c:\Users\ST000082\Documents\Codes\myosuite\myosuite\envs\myo\myochallenge\..\assets\leg\sample_gait_cycle.csv",
        "reset_type": "random",
        "terrain": "flat",
        "max_episode_steps": 1000,
    }
)