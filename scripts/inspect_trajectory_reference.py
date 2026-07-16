import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Définition du chemin ---
# On reconstruit le chemin de manière robuste
curr_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
csv_path = os.path.normpath(os.path.join(curr_dir, '../myosuite/envs/myo/assets/leg/sample_gait_cycle.csv'))

print(f"🔍 Chargement du fichier : {csv_path}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Le fichier CSV n'a pas été trouvé à l'emplacement : {csv_path}")

# --- 2. Lecture du CSV ---
df = pd.read_csv(csv_path)

# --- 3. Analyse structurelle de base ---
print("\n" + "="*40)
print("📊 DIAGNOSTIC GENERAL DU CSV")
print("="*40)
print(f"• Nombre de frames (lignes) : {len(df)}")
print(f"• Nombre de variables (colonnes) : {len(df.columns)}")
print(f"• Liste des colonnes détectées :\n  {list(df.columns)}")

# --- 4. Vérification du temps (Time step) ---
# Souvent, la première colonne s'appelle 'time' ou 't'. Sinon on utilise l'index.
time_col = None
for col in ['time', 't', 'Time', 'T']:
    if col in df.columns:
        time_col = col
        break

if time_col:
    time_diffs = np.diff(df[time_col].values)
    mean_dt = np.mean(time_diffs)
    std_dt = np.std(time_diffs)
    print(f"\n⏱️ ANALYSE TEMPORELLE (colonne '{time_col}') :")
    print(f"  - Durée totale de la trajectoire : {df[time_col].iloc[-1] - df[time_col].iloc[0]:.3f} secondes")
    print(f"  - Pas de temps moyen (dt) : {mean_dt:.5f}s (Fréquence : {1/mean_dt:.1f} Hz)")
    print(f"  - Fluctuation du dt (std) : {std_dt:.6f}s (si proche de 0, le pas de temps est constant)")
else:
    print("\n⚠️ ATTENTION : Aucune colonne de temps détectée (type 'time' ou 't'). L'index sera utilisé.")

# --- 5. Analyse des statistiques des articulations ---
print("\n📐 PLAGE DE VALEURS DES ARTICULATIONS (Exemples) :")
# On ignore 'time' ou les coordonnées globales comme pelvis_tx, pelvis_ty, pelvis_tz pour l'affichage des angles
exclude_cols = [time_col, 'time', 't', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
joint_cols = [c for c in df.columns if c not in exclude_cols]

stats = df[joint_cols].describe().loc[['min', 'max', 'mean']]
print(stats.to_string())

# --- 6. Vérification des unités (Radians vs Degrés) ---
# Les modèles MuJoCo utilisent exclusivement des RADIANS pour les joints revolute.
# Si tes angles dépassent largement [-3.14, 3.14] (ex: des valeurs de 45.0, -90.0), ils sont en degrés !
max_abs_val = df[joint_cols].abs().max().max()
print("\n🔮 DETECTEUR D'UNITÉS :")
if max_abs_val > 2 * np.pi:
    print(f"  🚨 ALERTE : La valeur absolue max est de {max_abs_val:.2f}. Tes données semblent être en DEGRÉS.")
    print("  👉 Tu devras les convertir en radians (rad = deg * pi / 180) avant de les envoyer à MuJoCo !")
else:
    print(f"  ✅ OK : La valeur absolue max est de {max_abs_val:.2f} (< 2*pi). Vos données semblent être en RADIANS.")

# --- 7. Visualisation Graphique ---
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

# Séparation des colonnes gauche/droite
joints_left = [c for c in joint_cols if c.endswith('_l') and any(x in c.lower() for x in ['hip', 'knee', 'ankle'])]
joints_right = [c for c in joint_cols if c.endswith('_r') and any(x in c.lower() for x in ['hip', 'knee', 'ankle'])]

# Subplot GAUCHE
for col in joints_left:
    ax_left.plot(df[time_col].values if time_col else df.index, df[col].values, label=col, alpha=0.8)
ax_left.set_title("Côté Gauche")
ax_left.set_xlabel("Temps (s)" if time_col else "Frames")
ax_left.set_ylabel("Angle (rad ou deg)")
ax_left.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_left.grid(True, linestyle='--', alpha=0.5)

# Subplot DROIT
for col in joints_right:
    ax_right.plot(df[time_col].values if time_col else df.index, df[col].values, label=col, alpha=0.8)
ax_right.set_title("Côté Droit")
ax_right.set_xlabel("Temps (s)" if time_col else "Frames")
ax_right.set_ylabel("Angle (rad ou deg)")
ax_right.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_right.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()