import numpy as np
from feature.drl.training.entities import TrainingState

def update_progress(logger, name, step, total, suffix):
    should = (step == total) or (step % max(1, total//25) == 0)
    if should:
        logger.progress(name, step, total, suffix=suffix)


def compute_mean_loss(losses):
    return float(np.mean(losses)) if losses else 0.0


def build_training_metadata(state: TrainingState):
    return {
        "completed_episodes": int(state.completed_episodes),
        "best_reward": float(state.best_reward),
    }
