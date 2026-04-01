from __future__ import annotations

from pathlib import Path
import numpy as np
from feature.drl.agent.agent import DQNAgent
from feature.drl.env.env import TradingEnv
from feature.drl.observation.processor import ObservationProcessor
from feature.drl.training.config import TrainingConfig
from feature.drl.training.entities import EpisodeResult, TrainingState
from feature.drl.training.logger import TrainingLogger
from feature.drl.training import utils


class Train:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.logger = TrainingLogger(width=self.config.progress_bar_width)
        self.processor = ObservationProcessor()

    def run(self) -> Path:
        df, env, agent, training_state = self._init_components()
        episode_count = max(1, int(self.config.episodes))
        for ep in range(episode_count):
            result = self._run_episode(agent, env, df, ep, episode_count)
            training_state = self._update_training_state(
                agent=agent,
                training_state=training_state,
                episode_result=result,
            )
        path = agent.save(
            self.config.model_path,
            metadata=utils.build_training_metadata(training_state),
        )
        self.logger.info(f"Model saved to {path}")
        return path


    def _init_components(self):
        from feature.processing.usecase.processing_uc import ProcessingUC
        df = ProcessingUC().run(tf=self.config.train_timeframe)
        env = TradingEnv(df)
        state = self.processor.transform(env.reset())
        agent = DQNAgent(state_size=len(state), action_size=env.action_space)
        training_state = self._load_checkpoint_if_available(agent)
        return df, env, agent, training_state

    def _run_episode(self, agent, env, df, ep, total_ep):
        state = self.processor.transform(env.reset())
        total_reward = 0.0
        losses = []
        max_steps = self.config.max_steps_per_episode or max(1, len(df) - env.current_step)
        for step in range(max_steps):
            state, reward, loss, done = self._train_step(agent, env, state)
            total_reward += reward
            if loss is not None:
                losses.append(loss)
            suffix = (
                f"episode={ep + 1}/{total_ep} "
                f"reward={total_reward:.4f} "
                f"eps={agent.epsilon:.3f}"
            )
            utils.update_progress(self.logger, "Training", step + 1, max_steps, suffix)
            if done:
                break
        mean_loss = utils.compute_mean_loss(losses)
        self.logger.info(
            f"Episode {ep + 1}/{total_ep} complete: "
            f"reward={total_reward:.4f}, loss={mean_loss:.6f}, epsilon={agent.epsilon:.3f}"
        )
        return EpisodeResult(
            total_reward=float(total_reward),
            mean_loss=float(mean_loss),
            epsilon=float(agent.epsilon),
        )

    def _train_step(self, agent, env, state):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state_t = self.processor.transform(next_state)
        agent.remember(state, action, reward, next_state_t, done)
        loss = agent.train_step()
        return next_state_t, reward, loss, done

    def _load_checkpoint_if_available(self, agent: DQNAgent) -> TrainingState:
        model_path = self.config.model_path
        if not self.config.resume_if_exists or not model_path.exists():
            self.logger.info(f"Starting new training run. Model path: {model_path}")
            return TrainingState()

        checkpoint = agent.load(model_path, strict=False)
        metadata = checkpoint.get("metadata") or {}
        training_state = TrainingState(
            completed_episodes=int(metadata.get("completed_episodes", 0)),
            best_reward=float(metadata.get("best_reward", float("-inf"))),
        )
        self.logger.info(
            f"Loaded checkpoint from {model_path} "
            f"(episodes={training_state.completed_episodes}, epsilon={agent.epsilon:.3f})"
        )
        return training_state

    def _update_training_state(
        self,
        agent: DQNAgent,
        training_state: TrainingState,
        episode_result: EpisodeResult,
    ) -> TrainingState:
        next_state = TrainingState(
            completed_episodes=training_state.completed_episodes + 1,
            best_reward=max(training_state.best_reward, episode_result.total_reward),
        )
        if episode_result.total_reward >= training_state.best_reward:
            best_path = agent.save(
                self.config.best_model_path,
                metadata=utils.build_training_metadata(next_state),
            )
            self.logger.info(
                f"Updated best checkpoint: {best_path} "
                f"(reward={episode_result.total_reward:.4f})"
            )
        return next_state
