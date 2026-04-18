import wandb
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


def setup_logging(
    project_name: str,
    run_name: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    offline: bool = False
) -> None:
    if offline:
        os.environ['WANDB_MODE'] = 'offline'
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity,
            reinit=True,
            settings=wandb.Settings(init_timeout=120)
        )
    except Exception:
        logging.warning("wandb online init failed; falling back to offline mode.")
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity,
            reinit=True
        )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True
) -> None:
    wandb.log(metrics, step=step, commit=commit)


def save_results_to_json(
    results: Dict[str, Any],
    save_path: str
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results_from_json(load_path: str) -> Dict[str, Any]:
    with open(load_path, 'r') as f:
        return json.load(f)
