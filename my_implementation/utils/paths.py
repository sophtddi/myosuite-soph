import os
import glob
from pathlib import Path

def _repo_root():
    return Path(__file__).resolve().parents[2]  # remonte jusqu'à myosuite-soph/

def get_results_root(subfolder="05-PredictiveSimulations/01-MyoSuite/results"):
    """Checkpoints & eval_logs : Directement sur le OneDrive de la VUB si dispo, sinon repli local."""
    
    # Path.home() remplacera dynamiquement "C:\\Users\\ST000082" ou "C:\\Users\\TonAutreNom"
    vub_onedrive = Path.home() / "OneDrive - Vrije Universiteit Brussel" / subfolder
    
    if vub_onedrive.exists() or vub_onedrive.parent.parent.exists():
        root = vub_onedrive
        # print(f"[Paths] ☁️ Enregistrement sur le OneDrive VUB : {root}")
    else:
        # Fallback local ultra-sécurisé au cas où l'autre PC n'a pas configuré ce OneDrive d'entreprise
        root = _repo_root() / "outputs_local"
        # print(f"[Paths] 💻 OneDrive VUB non trouvé. Repli sur le dossier local : {root}")
        
    root.mkdir(parents=True, exist_ok=True)
    return root

def get_local_logs_root():
    """tb_logs : toujours local, jamais sur OneDrive (évite le surcoût de sync)."""
    root = _repo_root() / "outputs_local" / "tb_logs"
    root.mkdir(parents=True, exist_ok=True)
    return root

def get_videos_dir():
    root = get_results_root()
    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    return videos_dir

def list_available_runs(eval_logs_dir):
    """Liste tous les runs pour lesquels un fichier evaluations.npz existe."""
    runs = []
    for path in glob.glob(str(Path(eval_logs_dir) / "*" / "evaluations.npz")):
        runs.append(os.path.basename(os.path.dirname(path)))
    return sorted(runs)