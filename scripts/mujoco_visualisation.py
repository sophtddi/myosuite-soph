import os
import time
import pandas as pd
import numpy as np
import gymnasium as gym
import my_implementation
import mujoco
import myosuite

VERTICAL_OFFSET = 0.0
# 1. Chargement des données du CSV
curr_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
csv_path = os.path.normpath(os.path.join(curr_dir, '../myosuite/envs/myo/assets/leg/sample_gait_cycle.csv'))
df = pd.read_csv(csv_path)

# 2. Initialisation de l'environnement
env_name = "myoChallengeOslRunFixed-v0"
print(f"🤖 Initialisation de l'environnement : {env_name}")

# On retire 'render_mode' du gym.make pour éviter les warnings s'il n'est pas supporté en natif
env = gym.make(env_name)
env.reset()

# Utilisation de .unwrapped pour contourner les wrappers de Gymnasium (comme TimeLimit)
unwrapped_env = env.unwrapped

# Extraction sécurisée de model et data (gère les anciennes et nouvelles versions de Myosuite)
if hasattr(unwrapped_env, 'mj_model'):  # Version récente (MuJoCo natif)
    model = unwrapped_env.mj_model
    data = unwrapped_env.mj_data
elif hasattr(unwrapped_env, 'sim'):      # Ancienne version (PyMjCF / mujoco-py)
    model = unwrapped_env.sim.model
    data = unwrapped_env.sim.data
else:
    raise AttributeError("Impossible de trouver les structures MuJoCo (sim ou mj_model) dans l'environnement.")

# 3. Récupération des noms d'articulations (joints)
# En MuJoCo natif, on extrait les noms via l'API de base
num_joints = model.njnt
mj_joint_names = [model.joint(i).name for i in range(num_joints)]
print(f"🔗 Articulations détectées dans MuJoCo ({len(mj_joint_names)}) : {mj_joint_names}")

# 4. Association des colonnes du CSV aux joints MuJoCo
joint_mapping = {}
for col in df.columns:
    if col in mj_joint_names:
        joint_mapping[col] = col

print(f"✅ Colonnes mappées avec succès ({len(joint_mapping)}/{len(df.columns)})")

# 5. Boucle de visualisation cinématique avec le viewer passif natif de MuJoCo
print("🚀 Lancement du visualiseur passif de MuJoCo...")
print("Une fenêtre graphique MuJoCo va s'ouvrir. Ferme-la pour quitter.")

try:
    # On lance le viewer passif officiel de DeepMind Mujoco (ultra stable et performant)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            # On dit à la caméra de SUIVRE un objet (TRACKING)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            
            # On cible le corps du pelvis (le tronc du modèle de jambe)
            # Remplace "pelvis" par le nom exact du corps principal si nécessaire
            viewer.cam.trackbodyid = model.body("pelvis").id
            
            # On règle la distance et les angles de suivi
            viewer.cam.distance = 3.5
            viewer.cam.azimuth = 90.0   # Profil parfait pour bien analyser la marche !
            viewer.cam.elevation = -10.0
        while viewer.is_running():
            for idx, row in df.iterrows():
                # Mise à jour des angles d'articulations dans la structure qpos
                for col in joint_mapping.keys():
                    try:
                        # Recherche de l'adresse (index) du joint dans qpos
                        joint_id = model.joint(col).id
                        qpos_adr = model.jnt_qposadr[joint_id]
                        data.qpos[qpos_adr] = row[col]
                    except Exception as e:
                        continue
                # 2. On ajuste la hauteur du pelvis (l'axe Z global)
                # Dans Myosuite, le pelvis est souvent un "freejoint" (7 coordonnées dans qpos)
                # Les 3 premières coordonnées [0, 1, 2] sont la position X, Y, Z dans l'espace.
                # data.qpos[2] contrôle la hauteur globale (Z).
                try:
                    # On applique la hauteur de la référence en lui ajoutant notre offset de sécurité
                    # Note : Si ton CSV n'a pas de colonne pelvis_tz brute, on force une hauteur minimale
                    if 'pelvis_tz' in df.columns:
                        data.qpos[2] = row['pelvis_tz'] + VERTICAL_OFFSET
                    else:
                        # Si pas de hauteur spécifiée, on triche en fixant une hauteur stable (ex: 0.85m)
                        data.qpos[2] = 0.88 + VERTICAL_OFFSET
                except Exception as e:
                    print(f"Erreur ajustement hauteur : {e}")
                
                # Mise à jour de la physique (cinématique directe)
                mujoco.mj_forward(model, data)
                
                # Rafraîchissement de la fenêtre
                viewer.sync()
                
                # Fréquence calée sur le pas de temps ou 25 FPS (0.04s)
                time.sleep(0.04)
                
except KeyboardInterrupt:
    print("\nVisualisation interrompue par l'utilisateur.")
finally:
    env.close()
    print("Fermeture de l'environnement.")