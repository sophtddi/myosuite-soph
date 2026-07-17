import os
import sys

# Ajoute le chemin racine de ton projet au PYTHONPATH si nécessaire
curr_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(os.path.normpath(os.path.join(curr_dir, "../")))

# Import de ta fonction d'entraînement
# Adapte l'import selon l'arborescence exacte de ton projet
from myosuite.my_implementation.training.trainer import train_model_imitation 

if __name__ == "__main__":
    print("🚦 DÉMARRAGE DU SMOKE TEST — RUN TRACK IMITATION 🚦")
    
    # 1. On définit des clés de reward conformes à ta nouvelle structure de classe
    test_reward_keys = {
        "sparse": 0,
        "solved": 10,
        "qpos_imitation": 30,
        "forward_bounded": 10,
        "alive_bonus": 15,
        "fall_penalty": -50,
    }
    
    # 2. Architecture du réseau (Identique à ta cible pour valider l'allocation mémoire)
    test_net_arch = dict(pi=[256, 256], vf=[256, 256])
    
    # 3. Paramètres de test très réduits
    # On utilise seulement 2 ou 4 processus pour le test afin d'accélérer l'init sur Windows
    TEST_ENVS = 4 
    TEST_TIMESTEPS = 10_000 
    
    print(f"• Environnements parallèles : {TEST_ENVS}")
    print(f"• Nombre total de timesteps : {TEST_TIMESTEPS}")
    print(f"• Distribution des rewards : {test_reward_keys}\n")

    try:
        # Lancement de l'entraînement court
        model = train_model_imitation(
            run_name="smoke_test_run3",
            weighted_reward_keys=test_reward_keys,
            total_timesteps=TEST_TIMESTEPS,
            seed=42,
            reset_type="random",
            net_arch=test_net_arch,
            n_envs=TEST_ENVS,
            resume_from=None,
        )
        print("\n✅ SMOKE TEST TERMINÉ AVEC SUCCÈS !")
        print("Les fichiers de checkpoints et logs ont été générés dans tes répertoires locaux.")
        
    except Exception as e:
        print(f"\n❌ LE SMOKE TEST A ÉCHOUÉ AVEC L'ERREUR SUIVANTE :", file=sys.stderr)
        import traceback
        traceback.print_exc()