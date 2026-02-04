"""
================================================================================
SMARTFLEET: MAPPO TRAINING MODULE
================================================================================
Multi-Agent Proximal Policy Optimization (MAPPO) training for the SmartFleet
delivery system with SUMO integration.

Features:
- Centralized Training, Decentralized Execution (CTDE)
- Parameter sharing across agents (optional)
- Proper multi-agent reward shaping
- Integration with Ray RLlib
- Comprehensive logging and visualization
- Checkpoint management

Author: Maram Ouelhazi
Project: IEEE RAS x VTS SmartFleet Challenge
================================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
import json
import time
from datetime import datetime
from collections import defaultdict
import argparse

# ==================== CONFIGURATION ====================

@dataclass
class MAPPOConfig:
    """MAPPO Training Configuration"""
    # Environment
    n_agents: int = 5
    use_sumo: bool = True
    sim_duration: int = 600
    
    # Training
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    
    # Batch sizes
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10
    
    # Multi-agent specific
    use_centralized_critic: bool = True
    share_policy: bool = True  # Parameter sharing
    
    # Logging
    checkpoint_freq: int = 10
    log_interval: int = 5
    save_path: str = "./smartfleet_checkpoints"
    
    # Hardware
    num_workers: int = 2
    num_gpus: float = 0  # Set to 1 if GPU available


# ==================== ENVIRONMENT WRAPPER FOR RLLIB ====================

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    HAS_MA_ENV = True
except ImportError:
    HAS_MA_ENV = False
    MultiAgentEnv = gym.Env  # Fallback


class SmartFleetMultiAgentEnv(MultiAgentEnv if HAS_MA_ENV else gym.Env):
    """
    RLlib-compatible wrapper for the SmartFleet SUMO environment.
    
    This wrapper:
    1. Converts the environment to RLlib's MultiAgentEnv format
    2. Provides global state for centralized critic
    3. Handles agent ID mapping
    4. Clips observations to valid bounds
    """
    
    def __init__(self, env_config: Dict = None):
        env_config = env_config or {}
        self.n_agents = env_config.get('n_agents', 5)
        self.use_sumo = env_config.get('use_sumo', False)  # Default False for training
        
        # Set up agent IDs FIRST (before super().__init__ which may call possible_agents)
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self._agent_ids = set(self.agent_ids)
        self._agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        
        # Now call parent init
        super().__init__()
        
        # Import the SUMO environment
        try:
            from delivery_agent_sumo import (
                MultiAgentDeliveryEnvSUMO, SimConfig, SUMO_AVAILABLE
            )
            
            config = SimConfig(
                sim_duration=env_config.get('sim_duration', 600),
                task_spawn_rate=env_config.get('task_spawn_rate', 0.2),
                verbose=env_config.get('verbose', False),
                log_every_n_steps=100,
                use_sumo_gui=False  # Never use GUI during training
            )
            
            self.env = MultiAgentDeliveryEnvSUMO(
                config=config,
                n_agents=self.n_agents,
                use_sumo=self.use_sumo and SUMO_AVAILABLE
            )
            self.using_sumo_env = True
            print(f"✓ Loaded SUMO environment (SUMO active: {self.use_sumo and SUMO_AVAILABLE})")
            
        except ImportError as e:
            print(f"⚠ Could not import SUMO environment: {e}")
            print("  Using standalone simulation environment")
            from delivery_agent import MultiAgentDeliveryEnv, SimConfig
            
            config = SimConfig(
                sim_duration=env_config.get('sim_duration', 600),
                task_spawn_rate=env_config.get('task_spawn_rate', 0.2),
            )
            self.env = MultiAgentDeliveryEnv(config=config, n_agents=self.n_agents)
            self.using_sumo_env = False
        
        # Single agent spaces (for reference)
        self._obs_space = self.env.observation_space
        self._act_space = self.env.action_space
        
        # Multi-agent spaces (required by Ray 2.53+)
        self.observation_spaces = {aid: self._obs_space for aid in self.agent_ids}
        self.action_spaces = {aid: self._act_space for aid in self.agent_ids}
        
        # Keep singular versions for compatibility
        self.observation_space = self._obs_space
        self.action_space = self._act_space
        
        # Global state dimension for centralized critic
        # Global state = concatenation of all agent observations + global metrics
        self.global_state_dim = self._obs_space.shape[0] * self.n_agents + 10
        
        # Episode tracking
        self.current_step = 0
        self.episode_rewards = defaultdict(float)
        
        # Clipping stats for debugging
        self.clip_count = 0
        self.total_obs_count = 0
    
    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Clip observation to valid bounds [-1, 1].
        Tracks clipping statistics for debugging.
        """
        self.total_obs_count += 1
        
        # Check if clipping is needed
        if np.any(obs < -1.0) or np.any(obs > 1.0):
            self.clip_count += 1
            # Uncomment for debugging:
            # if self.clip_count % 100 == 0:
            #     print(f"⚠ Clipping obs: min={obs.min():.2f}, max={obs.max():.2f} "
            #           f"(clipped {self.clip_count}/{self.total_obs_count})")
        
        return np.clip(obs, -1.0, 1.0).astype(np.float32)
    
    def get_agent_ids(self):
        """Return list of agent IDs (required by Ray 2.53+)"""
        return self._agent_ids
    
    @property
    def possible_agents(self):
        """Return all possible agent IDs"""
        return self.agent_ids
    
    def reset(self, *, seed=None, options=None):
        """Reset environment and return initial observations"""
        observations, info = self.env.reset(seed=seed)
        
        self.current_step = 0
        self.episode_rewards = defaultdict(float)
        
        # Reset clipping stats each episode
        self.clip_count = 0
        self.total_obs_count = 0
        
        # Convert to multi-agent format with clipping
        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = self._clip_obs(observations[i])
        
        return obs_dict, info
    
    def step(self, action_dict: Dict[str, int]):
        """Execute actions for all agents"""
        # Convert action dict to list
        actions = [action_dict.get(aid, 14) for aid in self.agent_ids]  # Default: CONTINUE
        
        # Step environment
        observations, rewards, done, truncated, infos = self.env.step(actions)
        
        self.current_step += 1
        
        # Convert to multi-agent format with clipping
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {}
        info_dict = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = self._clip_obs(observations[i])
            reward_dict[agent_id] = rewards[i]
            terminated_dict[agent_id] = done
            truncated_dict[agent_id] = truncated
            info_dict[agent_id] = infos[i] if i < len(infos) else {}
            
            self.episode_rewards[agent_id] += rewards[i]
        
        # Add __all__ key for RLlib
        terminated_dict["__all__"] = done
        truncated_dict["__all__"] = truncated
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
    
    def get_global_state(self) -> np.ndarray:
        """
        Get global state for centralized critic.
        Includes all agent observations + global metrics.
        """
        global_state = []
        
        # Concatenate all agent observations (with clipping)
        for agent in self.env.agents:
            visible_tasks = self.env._get_visible_tasks(agent)
            nearby_agents = self.env._get_nearby_agents(agent)
            obs = agent.get_observation(self.env.current_time, visible_tasks, nearby_agents)
            global_state.append(self._clip_obs(obs))
        
        # Add global metrics (already normalized)
        metrics = self.env.get_metrics()
        global_metrics = np.array([
            metrics['total_completed'] / 100.0,
            metrics['on_time_rate'],
            metrics['total_pending'] / 50.0,
            metrics['total_energy_kwh'] / 1000.0,
            metrics['fairness_variance'] / 10.0,
            metrics['active_agents'] / self.n_agents,
            self.env.current_time / self.env.config.sim_duration,
            len(self.env.tasks) / 100.0,
            len(self.env.completed_tasks) / 100.0,
            len(self.env.cancelled_tasks) / 50.0,
        ], dtype=np.float32)
        
        # Clip global metrics too
        global_state.append(np.clip(global_metrics, -1.0, 1.0))
        
        return np.concatenate(global_state)
    
    def get_metrics(self) -> Dict:
        """Get environment metrics"""
        return self.env.get_metrics()
    
    def get_clip_stats(self) -> Dict:
        """Get observation clipping statistics"""
        return {
            'clip_count': self.clip_count,
            'total_obs_count': self.total_obs_count,
            'clip_rate': self.clip_count / max(1, self.total_obs_count)
        }
    
    def close(self):
        """Clean up"""
        if hasattr(self.env, 'close'):
            self.env.close()


# ==================== CUSTOM CALLBACKS ====================

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.env import BaseEnv
    from ray.rllib.evaluation import RolloutWorker
    from ray.rllib.policy import Policy
    # Episode moved in Ray 2.x
    try:
        from ray.rllib.evaluation import Episode
    except ImportError:
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode as Episode
    RLLIB_AVAILABLE = True
except ImportError as e:
    RLLIB_AVAILABLE = False
    print(f"⚠ Ray RLlib not available: {e}")


if RLLIB_AVAILABLE:
    class SmartFleetCallbacks(DefaultCallbacks):
        """Custom callbacks for logging SmartFleet-specific metrics"""
        
        def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
            """Initialize episode metrics"""
            episode.user_data["deliveries_completed"] = 0
            episode.user_data["on_time_deliveries"] = 0
            episode.user_data["total_energy"] = 0
            episode.user_data["agent_rewards"] = defaultdict(float)
        
        def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
            """Track metrics during episode"""
            # Get the underlying environment
            env = base_env.get_sub_environments()[0]
            if hasattr(env, 'env'):
                real_env = env.env
                if hasattr(real_env, 'completed_tasks'):
                    episode.user_data["deliveries_completed"] = len(real_env.completed_tasks)
                    episode.user_data["on_time_deliveries"] = sum(
                        1 for t in real_env.completed_tasks 
                        if t.delivery_time and t.delivery_time <= t.deadline
                    )
        
        def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
            """Log final episode metrics"""
            env = base_env.get_sub_environments()[0]
            
            if hasattr(env, 'get_metrics'):
                metrics = env.get_metrics()
                
                episode.custom_metrics["deliveries_completed"] = metrics.get('total_completed', 0)
                episode.custom_metrics["on_time_rate"] = metrics.get('on_time_rate', 0)
                episode.custom_metrics["energy_per_delivery"] = metrics.get('energy_per_delivery', 0)
                episode.custom_metrics["total_distance_km"] = metrics.get('total_distance_km', 0)
                episode.custom_metrics["fairness_variance"] = metrics.get('fairness_variance', 0)
                episode.custom_metrics["pending_tasks"] = metrics.get('total_pending', 0)


# ==================== MAPPO TRAINER ====================

class MAPPOTrainer:
    """
    MAPPO Trainer for SmartFleet using Ray RLlib.
    
    Implements:
    - Centralized Training, Decentralized Execution (CTDE)
    - Parameter sharing (optional)
    - Multi-agent coordination rewards
    """
    
    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.training_iteration = 0
        self.best_reward = float('-inf')
        self.training_history = []
        
        # Create save directory
        os.makedirs(config.save_path, exist_ok=True)
        
        if not RLLIB_AVAILABLE:
            raise ImportError(
                "Ray RLlib is required for MAPPO training. "
                "Install with: pip install 'ray[rllib]'"
            )
        
        # Initialize Ray
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=config.num_workers + 2)
        
        # Build algorithm
        self.algo = self._build_algorithm()
    
    def _build_algorithm(self):
        """Build RLlib PPO algorithm with multi-agent config (Ray 2.5+)"""
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.policy.policy import PolicySpec

        # Environment config
        env_config = {
            "n_agents": self.config.n_agents,
            "use_sumo": self.config.use_sumo,
            "sim_duration": self.config.sim_duration,
            "task_spawn_rate": 0.2,
            "verbose": False,
        }

        # Create a test env to extract spaces
        test_env = SmartFleetMultiAgentEnv(env_config)
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        agent_ids = test_env.agent_ids
        test_env.close()

        # ---- Multi-agent policies ----
        if self.config.share_policy:
            policies = {
                "shared_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                )
            }
            policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"
            policies_to_train = ["shared_policy"]
        else:
            policies = {
                f"policy_{i}": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                )
                for i in range(self.config.n_agents)
            }
            policy_mapping_fn = (
                lambda agent_id, *args, **kwargs:
                f"policy_{agent_id.split('_')[1]}"
            )
            policies_to_train = list(policies.keys())

        # ---- PPO config (Ray 2.5+) ----
        config = (
            PPOConfig()
            # Keep old API stack (STABLE)
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .framework("torch")
            .environment(
                env=SmartFleetMultiAgentEnv,
                env_config=env_config,
            )

            # 🔥 REPLACEMENT FOR .rollouts()
            .env_runners(
                num_env_runners=self.config.num_workers,
                rollout_fragment_length="auto",
            )

            .training(
                gamma=self.config.gamma,
                lr=self.config.learning_rate,
                lambda_=self.config.gae_lambda,
                clip_param=self.config.clip_param,
                entropy_coeff=self.config.entropy_coeff,
                vf_loss_coeff=self.config.value_loss_coeff,
                grad_clip=self.config.max_grad_norm,
                train_batch_size=self.config.train_batch_size,
                minibatch_size=self.config.sgd_minibatch_size,
                num_sgd_iter=self.config.num_sgd_iter,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=policies_to_train,
            )
            .callbacks(SmartFleetCallbacks)
            .resources(num_gpus=self.config.num_gpus)
            .debugging(log_level="WARN")
        )

        return config.build()

    
    def train(self, n_iterations: int = None):
        """
        Run training loop.
        
        Args:
            n_iterations: Number of training iterations (default: calculated from total_timesteps)
        """
        if n_iterations is None:
            n_iterations = self.config.total_timesteps // self.config.train_batch_size
        
        print("\n" + "="*70)
        print("SMARTFLEET MAPPO TRAINING")
        print("="*70)
        print(f"  Agents: {self.config.n_agents}")
        print(f"  Policy sharing: {self.config.share_policy}")
        print(f"  Centralized critic: {self.config.use_centralized_critic}")
        print(f"  Total iterations: {n_iterations}")
        print(f"  Batch size: {self.config.train_batch_size}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for i in range(n_iterations):
            self.training_iteration = i + 1
            
            # Train one iteration
            result = self.algo.train()
            
            # Extract metrics
            metrics = self._extract_metrics(result)
            self.training_history.append(metrics)
            
            # Logging
            if (i + 1) % self.config.log_interval == 0:
                self._log_progress(metrics, start_time)
            
            # Checkpointing
            if (i + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(metrics)
            
            # Early stopping check (optional)
            if metrics.get('deliveries_completed', 0) >= 100 and metrics.get('on_time_rate', 0) >= 0.95:
                print(f"\n🎉 Reached target performance! Stopping early.")
                break
        
        # Final checkpoint
        self._save_checkpoint(metrics, final=True)
        
        # Training summary
        self._print_summary(start_time)
        
        return self.training_history
    
    def _extract_metrics(self, result: Dict) -> Dict:
        """Extract relevant metrics from training result"""
        custom = result.get('custom_metrics', {})
        
        return {
            'iteration': self.training_iteration,
            'timesteps_total': result.get('timesteps_total', 0),
            'episode_reward_mean': result.get('episode_reward_mean', 0),
            'episode_reward_max': result.get('episode_reward_max', 0),
            'episode_len_mean': result.get('episode_len_mean', 0),
            'deliveries_completed': custom.get('deliveries_completed_mean', 0),
            'on_time_rate': custom.get('on_time_rate_mean', 0),
            'energy_per_delivery': custom.get('energy_per_delivery_mean', 0),
            'total_distance_km': custom.get('total_distance_km_mean', 0),
            'policy_loss': result.get('info', {}).get('learner', {}).get(
                'shared_policy' if self.config.share_policy else 'policy_0', {}
            ).get('policy_loss', 0),
            'entropy': result.get('info', {}).get('learner', {}).get(
                'shared_policy' if self.config.share_policy else 'policy_0', {}
            ).get('entropy', 0),
        }
    
    def _log_progress(self, metrics: Dict, start_time: float):
        """Log training progress"""
        elapsed = time.time() - start_time
        
        print(f"\n[Iteration {metrics['iteration']}] "
              f"Time: {elapsed/60:.1f}min | "
              f"Steps: {metrics['timesteps_total']:,}")
        print(f"  Reward: {metrics['episode_reward_mean']:.2f} "
              f"(max: {metrics['episode_reward_max']:.2f})")
        print(f"  Deliveries: {metrics['deliveries_completed']:.1f} | "
              f"On-Time: {metrics['on_time_rate']:.1%}")
        print(f"  Energy/Delivery: {metrics['energy_per_delivery']:.3f} kWh | "
              f"Distance: {metrics['total_distance_km']:.2f} km")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f} | "
              f"Entropy: {metrics['entropy']:.4f}")
    
    def _save_checkpoint(self, metrics: Dict, final: bool = False):
        """Save training checkpoint"""
        checkpoint_path = self.algo.save(self.config.save_path)
        
        # Track best model
        if metrics['episode_reward_mean'] > self.best_reward:
            self.best_reward = metrics['episode_reward_mean']
            best_path = os.path.join(self.config.save_path, "best_model")
            self.algo.save(best_path)
            print(f"  ✓ New best model saved (reward: {self.best_reward:.2f})")
        
        # Save training history
        history_path = os.path.join(self.config.save_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        if final:
            print(f"\n✓ Final checkpoint saved to: {checkpoint_path}")
    
    def _print_summary(self, start_time: float):
        """Print training summary"""
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Total iterations: {self.training_iteration}")
        print(f"  Best reward: {self.best_reward:.2f}")
        
        if self.training_history:
            final = self.training_history[-1]
            print(f"\n  Final Performance:")
            print(f"    Deliveries: {final['deliveries_completed']:.1f}")
            print(f"    On-Time Rate: {final['on_time_rate']:.1%}")
            print(f"    Energy/Delivery: {final['energy_per_delivery']:.3f} kWh")
        
        print(f"\n  Checkpoints saved to: {self.config.save_path}")
        print("="*70)
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Dict:
        """
        Evaluate trained policy.
        
        Args:
            n_episodes: Number of evaluation episodes
            render: Whether to use SUMO GUI
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*70)
        print("EVALUATING TRAINED POLICY")
        print("="*70)
        
        # Create evaluation environment
        eval_config = {
            'n_agents': self.config.n_agents,
            'use_sumo': render,  # Only use SUMO if rendering
            'sim_duration': self.config.sim_duration,
            'verbose': True,
        }
        
        eval_env = SmartFleetMultiAgentEnv(eval_config)
        
        all_metrics = []
        total_rewards = []
        
        for ep in range(n_episodes):
            print(f"\nEpisode {ep + 1}/{n_episodes}")
            
            obs, _ = eval_env.reset(seed=ep)
            episode_reward = 0
            done = False
            
            while not done:
                # Get actions from policy
                actions = {}
                for agent_id in eval_env.agent_ids:
                    action = self.algo.compute_single_action(
                        obs[agent_id],
                        policy_id="shared_policy" if self.config.share_policy else f"policy_{agent_id.split('_')[1]}"
                    )
                    actions[agent_id] = action
                
                obs, rewards, terminateds, truncateds, infos = eval_env.step(actions)
                episode_reward += sum(rewards.values())
                done = terminateds.get("__all__", False)
            
            # Get final metrics
            metrics = eval_env.get_metrics()
            all_metrics.append(metrics)
            total_rewards.append(episode_reward)
            
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Deliveries: {metrics['total_completed']} | "
                  f"On-Time: {metrics['on_time_rate']:.1%}")
        
        eval_env.close()
        
        # Aggregate results
        avg_metrics = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_deliveries': np.mean([m['total_completed'] for m in all_metrics]),
            'mean_on_time_rate': np.mean([m['on_time_rate'] for m in all_metrics]),
            'mean_energy_per_delivery': np.mean([m['energy_per_delivery'] for m in all_metrics if m['energy_per_delivery'] > 0]),
        }
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"  Mean Reward: {avg_metrics['mean_reward']:.2f} ± {avg_metrics['std_reward']:.2f}")
        print(f"  Mean Deliveries: {avg_metrics['mean_deliveries']:.1f}")
        print(f"  Mean On-Time Rate: {avg_metrics['mean_on_time_rate']:.1%}")
        print(f"  Mean Energy/Delivery: {avg_metrics['mean_energy_per_delivery']:.3f} kWh")
        print("="*70)
        
        return avg_metrics
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint"""
        self.algo.restore(checkpoint_path)
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    
    def close(self):
        """Clean up resources"""
        self.algo.stop()
        import ray
        ray.shutdown()


# ==================== GREEDY BASELINE ====================

class GreedyBaseline:
    """
    Greedy heuristic baseline for comparison.
    Uses rule-based decision making without learning.
    """
    
    def __init__(self, config: MAPPOConfig):
        self.config = config
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """Run greedy baseline evaluation"""
        print("\n" + "="*70)
        print("GREEDY BASELINE EVALUATION")
        print("="*70)
        
        env_config = {
            'n_agents': self.config.n_agents,
            'use_sumo': False,
            'sim_duration': self.config.sim_duration,
            'verbose': False,
        }
        
        env = SmartFleetMultiAgentEnv(env_config)
        
        all_metrics = []
        total_rewards = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            episode_reward = 0
            done = False
            
            while not done:
                actions = {}
                for i, agent_id in enumerate(env.agent_ids):
                    action = self._greedy_action(env.env.agents[i], env.env)
                    actions[agent_id] = action
                
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                done = terminateds.get("__all__", False)
            
            metrics = env.get_metrics()
            all_metrics.append(metrics)
            total_rewards.append(episode_reward)
            
            print(f"Episode {ep + 1}: Reward={episode_reward:.2f}, "
                  f"Deliveries={metrics['total_completed']}, "
                  f"On-Time={metrics['on_time_rate']:.1%}")
        
        env.close()
        
        avg_metrics = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_deliveries': np.mean([m['total_completed'] for m in all_metrics]),
            'mean_on_time_rate': np.mean([m['on_time_rate'] for m in all_metrics]),
        }
        
        print("\n" + "-"*50)
        print("GREEDY BASELINE RESULTS:")
        print(f"  Mean Reward: {avg_metrics['mean_reward']:.2f} ± {avg_metrics['std_reward']:.2f}")
        print(f"  Mean Deliveries: {avg_metrics['mean_deliveries']:.1f}")
        print(f"  Mean On-Time Rate: {avg_metrics['mean_on_time_rate']:.1%}")
        print("-"*50)
        
        return avg_metrics
    
    def _greedy_action(self, agent, env) -> int:
        """Greedy action selection"""
        # Battery critical
        battery_percent = agent.battery / agent.spec.max_battery_kwh
        if battery_percent < 0.30 or agent.is_returning_to_depot:
            return 9  # GO_TO_DEPOT
        
        # Has task
        if agent.current_task:
            return 14  # CONTINUE
        
        # Try to accept task
        visible_tasks = env._get_visible_tasks(agent)
        check_fn = agent.can_accept_task if hasattr(agent, 'can_accept_task') else agent._can_accept_task
        acceptable = [t for t in visible_tasks if check_fn(t)]
        
        if acceptable:
            def score(task):
                dist = agent.position.distance_to(task.pickup_loc)
                return dist / (task.priority + 1)
            
            best = min(acceptable, key=score)
            idx = visible_tasks.index(best)
            return min(idx, 4)
        
        # Explore
        import random
        return random.choice([5, 6, 7, 8])


# ==================== VISUALIZATION ====================

def plot_training_curves(history_path: str, output_path: str = "training_curves.png"):
    """
    Plot training curves from saved history.
    
    Requires matplotlib: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    iterations = [h['iteration'] for h in history]
    rewards = [h['episode_reward_mean'] for h in history]
    deliveries = [h['deliveries_completed'] for h in history]
    on_time = [h['on_time_rate'] * 100 for h in history]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SmartFleet MAPPO Training Progress', fontsize=14)
    
    # Reward
    axes[0, 0].plot(iterations, rewards, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Mean Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Deliveries
    axes[0, 1].plot(iterations, deliveries, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Deliveries')
    axes[0, 1].set_title('Deliveries Completed')
    axes[0, 1].grid(True, alpha=0.3)
    
    # On-time rate
    axes[1, 0].plot(iterations, on_time, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('On-Time Rate (%)')
    axes[1, 0].set_title('On-Time Delivery Rate')
    axes[1, 0].axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Target')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Combined normalized
    rewards_norm = np.array(rewards) / max(rewards) if max(rewards) > 0 else rewards
    deliveries_norm = np.array(deliveries) / max(deliveries) if max(deliveries) > 0 else deliveries
    on_time_norm = np.array(on_time) / 100
    
    axes[1, 1].plot(iterations, rewards_norm, 'b-', label='Reward', linewidth=2)
    axes[1, 1].plot(iterations, deliveries_norm, 'g-', label='Deliveries', linewidth=2)
    axes[1, 1].plot(iterations, on_time_norm, 'r-', label='On-Time', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].set_title('All Metrics (Normalized)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Training curves saved to: {output_path}")
    plt.close()


# ==================== MAIN ====================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SmartFleet MAPPO Training')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'baseline', 'plot'],
                       help='Mode: train, evaluate, baseline, or plot')
    parser.add_argument('--n_agents', type=int, default=5,
                       help='Number of agents')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Training iterations')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for evaluation')
    parser.add_argument('--use_sumo', action='store_true',
                       help='Use SUMO during evaluation')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Configuration
    config = MAPPOConfig(
        n_agents=args.n_agents,
        use_sumo=False,  # Don't use SUMO during training
        total_timesteps=args.iterations * 4000,
        num_workers=2,
    )
    
    if args.mode == 'train':
        print("\n" + "="*70)
        print("SMARTFLEET MAPPO TRAINING")
        print("="*70)
        
        if not RLLIB_AVAILABLE:
            print("\n⚠ Ray RLlib not available!")
            print("  Install with: pip install 'ray[rllib]' torch")
            print("\n  Running greedy baseline instead...")
            baseline = GreedyBaseline(config)
            baseline.evaluate(n_episodes=5)
            return
        
        trainer = MAPPOTrainer(config)
        
        try:
            # Run greedy baseline first
            print("\n--- Running Greedy Baseline for Comparison ---")
            baseline = GreedyBaseline(config)
            baseline_results = baseline.evaluate(n_episodes=5)
            
            # Train MAPPO
            print("\n--- Starting MAPPO Training ---")
            trainer.train(n_iterations=args.iterations)
            
            # Evaluate
            print("\n--- Evaluating Trained Policy ---")
            eval_results = trainer.evaluate(n_episodes=args.episodes)
            
            # Compare
            print("\n" + "="*70)
            print("COMPARISON: MAPPO vs Greedy Baseline")
            print("="*70)
            print(f"  Metric              | MAPPO        | Greedy")
            print(f"  --------------------|--------------|------------")
            print(f"  Mean Reward         | {eval_results['mean_reward']:>12.2f} | {baseline_results['mean_reward']:>12.2f}")
            print(f"  Mean Deliveries     | {eval_results['mean_deliveries']:>12.1f} | {baseline_results['mean_deliveries']:>12.1f}")
            print(f"  Mean On-Time Rate   | {eval_results['mean_on_time_rate']:>11.1%} | {baseline_results['mean_on_time_rate']:>11.1%}")
            print("="*70)
            
            # Plot training curves
            history_path = os.path.join(config.save_path, "training_history.json")
            if os.path.exists(history_path):
                plot_training_curves(history_path)
            
        finally:
            trainer.close()
    
    elif args.mode == 'evaluate':
        if not RLLIB_AVAILABLE:
            print(" Ray RLlib required for evaluation")
            return
        
        trainer = MAPPOTrainer(config)
        
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        else:
            # Try to load best model
            best_path = os.path.join(config.save_path, "best_model")
            if os.path.exists(best_path):
                trainer.load_checkpoint(best_path)
            else:
                print(" No checkpoint found. Please specify --checkpoint")
                return
        
        trainer.evaluate(n_episodes=args.episodes, render=args.use_sumo)
        trainer.close()
    
    elif args.mode == 'baseline':
        baseline = GreedyBaseline(config)
        baseline.evaluate(n_episodes=args.episodes)
    
    elif args.mode == 'plot':
        history_path = os.path.join(config.save_path, "training_history.json")
        if os.path.exists(history_path):
            plot_training_curves(history_path)
        else:
            print(f" No training history found at: {history_path}")


if __name__ == "__main__":
    main()
