import os
import glob
from pathlib import Path

def _repo_root():
    return Path(__file__).resolve().parents[2]  # remonte jusqu'à myosuite-soph/

def get_results_root(subfolder="05-PredictiveSimulations/01-MyoSuite/results"):
    """Checkpoints & eval_logs : sur OneDrive si dispo, sinon repli local."""
    onedrive_root = os.environ.get("OneDriveCommercial") or os.environ.get("OneDrive")
    root = Path(onedrive_root) / subfolder if onedrive_root else _repo_root() / "outputs_local"
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