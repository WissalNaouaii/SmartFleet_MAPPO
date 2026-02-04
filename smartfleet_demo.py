"""
================================================================================
SMARTFLEET: QUICK DEMO (No Ray Required)
================================================================================
This script demonstrates the SmartFleet system without requiring Ray RLlib.
It runs the greedy baseline and shows expected metrics.

Run: python smartfleet_demo.py
================================================================================
"""

import numpy as np
import random
import time
from collections import defaultdict
import json
import os

# Import the SUMO environment
try:
    from delivery_agent_sumo import (
        MultiAgentDeliveryEnvSUMO, SimConfig, RLDeliveryAgentSUMO,
        VehicleType, TaskStatus, SUMO_AVAILABLE
    )
    HAS_ENV = True
    print("✓ Environment loaded successfully")
except ImportError as e:
    HAS_ENV = False
    print(f"✗ Could not import environment: {e}")


class SmartFleetDemo:
    """Quick demonstration of SmartFleet capabilities"""
    
    def __init__(self, n_agents: int = 5, use_sumo: bool = False):
        if not HAS_ENV:
            raise ImportError("Environment not available. Check delivery_agent_sumo.py exists.")
        
        self.n_agents = n_agents
        self.use_sumo = use_sumo
        
        # Create environment
        config = SimConfig(
            sim_duration=600,
            task_spawn_rate=0.2,
            verbose=False,  # Reduce spam
            log_every_n_steps=200,
            use_sumo_gui=False
        )
        
        self.env = MultiAgentDeliveryEnvSUMO(
            config=config, 
            n_agents=n_agents,
            use_sumo=use_sumo and SUMO_AVAILABLE
        )
        
        print(f"✓ Environment created with {n_agents} agents")
        print(f"  SUMO active: {use_sumo and SUMO_AVAILABLE}")
    
    def greedy_action(self, agent) -> int:
        """Simple greedy policy"""
        # Battery critical
        battery_pct = agent.battery / agent.spec.max_battery_kwh
        if battery_pct < 0.30 or agent.is_returning_to_depot:
            return 9  # GO_TO_DEPOT
        
        # Has task - continue
        if agent.current_task:
            return 14  # CONTINUE
        
        # Get visible tasks
        visible_tasks = self.env._get_visible_tasks(agent)
        
        # Check which tasks agent can accept
        acceptable = []
        for t in visible_tasks:
            if agent.can_accept_task(t):
                acceptable.append(t)
        
        if acceptable:
            # Pick closest high-priority task
            def score(task):
                dist = agent.position.distance_to(task.pickup_loc)
                return dist / (task.priority + 1)
            
            best = min(acceptable, key=score)
            idx = visible_tasks.index(best)
            return min(idx, 4)  # ACCEPT_TASK_0-4
        
        # Explore randomly
        return random.choice([5, 6, 7, 8])  # MOVE directions
    
    def run_episode(self, seed: int = 0, verbose: bool = True) -> dict:
        """Run a single episode with greedy policy"""
        observations, _ = self.env.reset(seed=seed)
        
        episode_reward = 0
        step = 0
        max_steps = 600  # Match sim_duration
        
        while step < max_steps:
            # Get greedy actions for all agents
            actions = []
            for agent in self.env.agents:
                action = self.greedy_action(agent)
                actions.append(action)
            
            # Step environment
            observations, rewards, done, truncated, infos = self.env.step(actions)
            episode_reward += sum(rewards)
            step += 1
            
            # Progress indicator
            if verbose and step % 100 == 0:
                metrics = self.env.get_metrics()
                print(f"  Step {step}: {metrics['total_completed']} deliveries, "
                      f"{metrics['on_time_rate']:.0%} on-time")
            
            if done:
                break
        
        # Get final metrics
        metrics = self.env.get_metrics()
        metrics['episode_reward'] = episode_reward
        metrics['steps'] = step
        
        if verbose:
            print(f"\n{'='*60}")
            print("EPISODE COMPLETE")
            print(f"{'='*60}")
            print(f"  Steps: {step}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Deliveries Completed: {metrics['total_completed']}")
            print(f"  On-Time Rate: {metrics['on_time_rate']:.1%}")
            print(f"  Energy per Delivery: {metrics['energy_per_delivery']:.3f} kWh")
            print(f"  Total Distance: {metrics['total_distance_km']:.2f} km")
            print(f"  Pending Tasks: {metrics['total_pending']}")
            print(f"\n  Per-Agent Performance:")
            for agent in self.env.agents:
                on_time_rate = agent.on_time_count / max(agent.completed_count, 1)
                print(f"    {agent.agent_id} ({agent.vehicle_type.value}): "
                      f"{agent.completed_count} deliveries, "
                      f"{on_time_rate:.0%} on-time")
        
        return metrics
    
    def run_benchmark(self, n_episodes: int = 3) -> dict:
        """Run multiple episodes and compute statistics"""
        print("\n" + "="*70)
        print("SMARTFLEET GREEDY BASELINE BENCHMARK")
        print("="*70)
        print(f"  Agents: {self.n_agents}")
        print(f"  Episodes: {n_episodes}")
        print("="*70)
        
        all_metrics = []
        
        for ep in range(n_episodes):
            print(f"\n--- Episode {ep + 1}/{n_episodes} ---")
            metrics = self.run_episode(seed=ep, verbose=True)
            all_metrics.append(metrics)
        
        # Aggregate results
        results = {
            'n_episodes': n_episodes,
            'mean_reward': np.mean([m['episode_reward'] for m in all_metrics]),
            'std_reward': np.std([m['episode_reward'] for m in all_metrics]),
            'mean_deliveries': np.mean([m['total_completed'] for m in all_metrics]),
            'std_deliveries': np.std([m['total_completed'] for m in all_metrics]),
            'mean_on_time_rate': np.mean([m['on_time_rate'] for m in all_metrics]),
            'mean_energy_per_delivery': np.mean([
                m['energy_per_delivery'] for m in all_metrics 
                if m['energy_per_delivery'] > 0
            ]) if any(m['energy_per_delivery'] > 0 for m in all_metrics) else 0,
            'mean_distance_km': np.mean([m['total_distance_km'] for m in all_metrics]),
        }
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(f"  Mean Reward:          {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Deliveries:      {results['mean_deliveries']:.1f} ± {results['std_deliveries']:.1f}")
        print(f"  Mean On-Time Rate:    {results['mean_on_time_rate']:.1%}")
        print(f"  Mean Energy/Delivery: {results['mean_energy_per_delivery']:.3f} kWh")
        print(f"  Mean Distance:        {results['mean_distance_km']:.2f} km")
        print("="*70)
        
        # What MAPPO should achieve
        print("\n" + "="*70)
        print("EXPECTED MAPPO IMPROVEMENTS")
        print("="*70)
        print("  After ~100 training iterations, MAPPO should achieve:")
        print(f"    Deliveries:    {results['mean_deliveries']:.0f} → {results['mean_deliveries']*2:.0f}+ (2x improvement)")
        print(f"    On-Time Rate:  {results['mean_on_time_rate']:.0%} → 90%+ ")
        print(f"    Energy/Del:    {results['mean_energy_per_delivery']:.2f} → {results['mean_energy_per_delivery']*0.7:.2f} kWh (30% reduction)")
        print("\n  Key MAPPO advantages over greedy:")
        print("    • Learns to coordinate between agents")
        print("    • Optimizes long-term rewards, not just immediate gains")
        print("    • Adapts to traffic patterns and task distributions")
        print("    • Balances workload across fleet (fairness)")
        print("="*70)
        
        return results
    
    def close(self):
        """Clean up"""
        if hasattr(self.env, 'close'):
            self.env.close()


def main():
    """Main demo entry point"""
    print("\n" + "="*70)
    print("SMARTFLEET MULTI-AGENT DELIVERY SYSTEM - DEMO")
    print("="*70)
    print("\nThis demo shows the greedy baseline performance.")
    print("Train with MAPPO to significantly improve these results!")
    print("\nTo train MAPPO:")
    print("  python smartfleet_mappo_training.py --mode train --iterations 50")
    print("="*70)
    
    # Run demo
    demo = SmartFleetDemo(n_agents=5, use_sumo=False)
    
    try:
        results = demo.run_benchmark(n_episodes=3)
        
        # Save results
        with open('demo_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to demo_results.json")
        
    finally:
        demo.close()
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
