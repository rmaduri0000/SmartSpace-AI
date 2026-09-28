"""Batch-train and save the SmartSpace AI DQN layout optimizer."""
import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SAMPLE_ROOMS
from engine.interior_env import InteriorEnv
from engine.dqn_agent import DQNAgent

def train_dqn(episodes: int = 100, max_steps_per_episode: int = 40, output_dir: str = "data/models"):
    if episodes < 1 or max_steps_per_episode < 1:
        raise ValueError("episodes and max_steps_per_episode must both be positive")
    print(f"=== Starting SmartSpace AI DQN Training ({episodes} episodes) ===")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment with Master Bedroom preset
    room_config = SAMPLE_ROOMS["master_bedroom"]
    env = InteriorEnv(room_config)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     model_path=str(output_path / "dqn_policy.npz"))
    
    history = {
        "episodes": [],
        "rewards": [],
        "ergonomics_scores": [],
        "collision_counts": [],
        "circulation_ratios": [],
        "loss": [],
        "training_updates": 0
    }
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        losses = []
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
                history["training_updates"] += 1
                
            state = next_state
            ep_reward += reward
            if done:
                break
                
        agent.update_target_network()
        
        final_metrics = env.evaluate_layout()
        avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
        
        history["episodes"].append(ep)
        history["rewards"].append(round(float(ep_reward), 2))
        history["ergonomics_scores"].append(round(float(final_metrics["ergonomics_score"]), 1))
        history["collision_counts"].append(final_metrics["collision_count"])
        history["circulation_ratios"].append(round(float(final_metrics["circulation_ratio"]), 2))
        history["loss"].append(round(avg_loss, 4))
        
        if ep % 10 == 0 or ep == episodes:
            print(f"Episode {ep:3d}/{episodes} | "
                  f"Reward: {ep_reward:7.2f} | "
                  f"Ergonomics: {final_metrics['ergonomics_score']:5.1f}% | "
                  f"Collisions: {final_metrics['collision_count']} | "
                  f"Circulation: {final_metrics['circulation_ratio']*100:4.0f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
                  
    if agent.is_trained:
        agent.save_model()
    history["real_training"] = agent.is_trained
    history["backend"] = agent.backend
    history["checkpoint"] = "dqn_policy.npz" if agent.is_trained else None

    history_file = output_path / "dqn_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"\nTraining completed! Checkpoint and telemetry saved to: {output_path}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartSpace AI DQN Layout Optimizer")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=30, help="Max steps per episode")
    args = parser.parse_args()
    
    train_dqn(episodes=args.episodes, max_steps_per_episode=args.steps)
