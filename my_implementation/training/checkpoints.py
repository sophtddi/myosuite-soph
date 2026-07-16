import glob
from pathlib import Path


def find_latest_checkpoint(run_name, checkpoints_dir):
    checkpoints_dir = Path(checkpoints_dir)
    pattern = str(checkpoints_dir / run_name / f"{run_name}_*_steps.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        for fallback in [checkpoints_dir / f"{run_name}_final.zip",
                          checkpoints_dir / run_name / "best_model.zip"]:
            if fallback.exists():
                return str(fallback)
        return None
    candidates.sort(key=lambda p: int(p.split("_")[-2]))
    return candidates[-1]