
import datetime
from pathlib import Path

def make_run_dir(model_name: str = "general"):
    """
    Create a timestamped run directory inside a model-specific experiments folder.

    Examples:
        make_run_dir("bert") -> experiments_bert/run-20251014-180300/
        make_run_dir("sgd")  -> experiments_sgd/run-20251014-180300/
        make_run_dir()       -> experiments_general/run-20251014-180300/
    """
    # Normalize model name
    model_name = model_name.lower().strip()
    root = f"experiments_{model_name}"

    # Create base folder (e.g., experiments_bert)
    Path(root).mkdir(parents=True, exist_ok=True)

    # Create timestamped subfolder
    run = datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    out = Path(root) / run
    out.mkdir(parents=True, exist_ok=True)

    return str(out)

