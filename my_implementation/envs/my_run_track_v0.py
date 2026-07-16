import numpy as np
from myosuite.envs.myo.myochallenge.run_track_v0 import RunTrack
from gymnasium.envs.registration import register
import mujoco


class RunTrackImitation(RunTrack):

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "sparse": 0,           # Neutralisé
        "solved": 10,
        "qpos_imitation": 30,  # Légèrement baissé pour faire de la place aux autres signaux
        "forward_bounded": 10, # Augmenté pour guider la progression
        "alive_bonus": 15,     # Nouveau : Récompense le maintien en position debout
        "fall_penalty": -50,   # Nouveau : Punition sévère si l'agent tombe
    }

    def _setup(self, **kwargs):
        self._imitation_index = 0
        self._target_velocity = 1.2
        self.fall_threshold = 0.65  # Hauteur du pelvis en dessous de laquelle on considère que l'agent est tombé

        super()._setup(**kwargs)

  
        csv_vel_y_idx = self.gait_cycle_headers.get('pelvis_vel_X', None)
        if csv_vel_y_idx is not None:
            self._target_velocity = np.abs(np.mean(self.INIT_DATA[:, csv_vel_y_idx]))
        else:
            self._target_velocity = 1.2  # Valeur de secours par défaut si la colonne manque

        print(f"[RunTrackImitation] Vitesse cible d'avancée calculée depuis le CSV : {self._target_velocity:.4f} m/s")

    def reset(self, **kwargs):
        # On commence à un index aléatoire dans les premiers 80% du cycle pour diversifier les états initiaux
        self._imitation_index = self.np_random.integers(0, int(self.INIT_DATA.shape[0] * 0.8))
        obs = super().reset(**kwargs)
        self._set_pose_from_reference(self._imitation_index)
        return obs

    def _set_pose_from_reference(self, ref_idx):
        for jnt in self.biological_jnt:
            if jnt in self.gait_cycle_headers:
                ref_qpos = self.INIT_DATA[ref_idx, self.gait_cycle_headers[jnt]]
                self.mj_data.joint(jnt).qpos[0] = ref_qpos
        
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def step(self, a, **kwargs):
        self._imitation_index += 1
        if self._imitation_index >= self.INIT_DATA.shape[0]:
            self._imitation_index = 0
        
        obs, reward, terminated, truncated, info = super().step(a, **kwargs)

        # --- Détection de la chute (Early Stopping) ---
        # Si le pelvis descend en dessous de la hauteur limite, l'épisode se termine immédiatement.
        # pelvis_z est généralement le 3ème élément (index 2) de qpos du freejoint du pelvis.
        pelvis_height = self.mj_data.qpos[2]
        
        if pelvis_height < self.fall_threshold:
            terminated = True
            # On passe l'information à l'info dict pour le tracking des métriques dans SB3
            info["fallen"] = True 
        else:
            info["fallen"] = False

        # On recalcule notre dictionnaire de reward customisé
        rwd_dict = self.get_reward_dict(obs)
        reward = rwd_dict["dense"]

        return obs, reward, terminated, truncated, info

    def _get_qpos_diff_array(self):
        diffs = []
        for jnt in self.biological_jnt:
            if jnt in self.gait_cycle_headers:
                ref_val = self.INIT_DATA[self._imitation_index, self.gait_cycle_headers[jnt]]
                cur_val = self.mj_data.joint(jnt).qpos[0]
                diffs.append(cur_val - ref_val)
        return np.array(diffs)

    def _get_forward_bounded_reward(self):
        # On récupère la vitesse linéaire de la racine (pelvis)
        current_vel = self.obs_dict["model_root_vel"].squeeze()[1]  # Vitesse le long de l'axe Y
        # Vitesse d'avancée réelle (positive quand on va vers les Y négatifs)
        forward_speed = -current_vel 
        speed_error = forward_speed - self._target_velocity
        return self.dt * np.exp(-5 * np.square(speed_error))

    def get_reward_dict(self, obs_dict):
        # 1. Calcul des composantes principales
        q_diff = self._get_qpos_diff_array()
        
        # qpos_imitation : Tolérance de 8 (standard). On multiplie par dt pour rester cohérent avec Myosuite
        qpos_imitation_value = self.dt * np.mean(np.exp(-8 * np.square(q_diff)))
        forward_bounded_value = self._get_forward_bounded_reward()

        # 2. Ajout des nouveaux signaux de survie et de chute
        pelvis_height = self.mj_data.qpos[2]
        is_fallen = pelvis_height < self.fall_threshold

        # Bonus de vie : l'agent gagne un montant fixe proportionnel au temps (dt) tant qu'il ne tombe pas
        alive_bonus_value = self.dt if not is_fallen else 0.0
        
        # Pénalité de chute : appliquée une seule fois si l'agent tombe
        fall_penalty_value = 1.0 if is_fallen else 0.0

        # 3. Récupération des clés par défaut de l'environnement parent (comme act_reg, pain, solved...)
        # On isole temporairement nos clés customisées
        custom_keys = ["qpos_imitation", "forward_bounded", "alive_bonus", "fall_penalty"]
        weights_to_restore = {}
        for key in custom_keys:
            w = self.rwd_keys_wt.pop(key, None)
            if w is not None:
                weights_to_restore[key] = w

        # Appel au parent pour calculer le "dense" de base (contenant pain, act_reg, solved...)
        rwd_dict = super().get_reward_dict(obs_dict)

        # Restauration des poids customisés dans l'environnement
        for key, w in weights_to_restore.items():
            self.rwd_keys_wt[key] = w

        # Injection des valeurs dans le dictionnaire de retour
        rwd_dict["qpos_imitation"] = qpos_imitation_value
        rwd_dict["forward_bounded"] = forward_bounded_value
        rwd_dict["alive_bonus"] = alive_bonus_value
        rwd_dict["fall_penalty"] = fall_penalty_value

        # 4. Assemblage final du reward "dense" (somme pondérée)
        rwd_dict["dense"] = (
            rwd_dict["dense"]
            + weights_to_restore.get("qpos_imitation", 0) * qpos_imitation_value
            + weights_to_restore.get("forward_bounded", 0) * forward_bounded_value
            + weights_to_restore.get("alive_bonus", 0) * alive_bonus_value
            + weights_to_restore.get("fall_penalty", 0) * fall_penalty_value
        )

        return rwd_dict