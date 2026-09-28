"""DQN layout optimizer with TensorFlow and trainable NumPy implementations."""
import copy
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReplayBuffer:
    """Fixed-size replay buffer for DQN transitions."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int32),
                np.asarray(rewards, dtype=np.float32), np.asarray(next_states, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN that persists checkpoints and has a real SGD NumPy fallback."""

    def __init__(self, state_dim: int = 133, action_dim: int = 48,
                 lr: float = 0.001, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995, model_path: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.memory = ReplayBuffer(capacity=5000)
        self.batch_size = 32
        default_path = PROJECT_ROOT / "data" / "models" / "dqn_policy.npz"
        requested_path = Path(model_path) if model_path else default_path
        self.model_path = requested_path if requested_path.is_absolute() else PROJECT_ROOT / requested_path

        self.q_net = self._build_model()
        self.target_net = self._build_model()
        self.is_trained = self._load_model()
        self.update_target_network()

    @property
    def backend(self) -> str:
        return "tensorflow" if TF_AVAILABLE else "numpy"

    def _build_model(self):
        if TF_AVAILABLE:
            model = keras.Sequential([
                layers.Input(shape=(self.state_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.action_dim, activation="linear"),
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss="huber")
            return model
        return SimpleNumpyMLP(self.state_dim, self.action_dim, learning_rate=self.lr)

    def _predict(self, model, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if TF_AVAILABLE and isinstance(model, keras.Model):
            return model.predict(states, verbose=0)
        return model.forward(states)

    def update_target_network(self) -> None:
        if TF_AVAILABLE and isinstance(self.q_net, keras.Model):
            self.target_net.set_weights(self.q_net.get_weights())
        else:
            self.target_net.copy_from(self.q_net)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        return int(np.argmax(self._predict(self.q_net, np.asarray(state)[None, :])[0]))

    def train_step(self) -> Optional[float]:
        """Apply a Bellman update from a sampled replay batch."""
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        targets = self._predict(self.q_net, states)
        next_values = np.max(self._predict(self.target_net, next_states), axis=1)
        targets[np.arange(self.batch_size), actions] = rewards + self.gamma * next_values * (~dones)

        if TF_AVAILABLE and isinstance(self.q_net, keras.Model):
            history = self.q_net.fit(states, targets, epochs=1, verbose=0)
            loss = float(history.history["loss"][0])
        else:
            loss = self.q_net.train_on_batch(states, targets)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.is_trained = True
        return float(loss)

    def save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        weights = self.q_net.get_weights() if TF_AVAILABLE and isinstance(self.q_net, keras.Model) else self.q_net.get_weights()
        np.savez(self.model_path, state_dim=self.state_dim, action_dim=self.action_dim,
                 backend=self.backend, trained=self.is_trained,
                 **{f"weight_{index}": value for index, value in enumerate(weights)})

    def _load_model(self) -> bool:
        if not self.model_path.is_file():
            return False
        try:
            with np.load(self.model_path, allow_pickle=False) as checkpoint:
                if int(checkpoint["state_dim"]) != self.state_dim or int(checkpoint["action_dim"]) != self.action_dim:
                    print(f"[DQN] Ignoring incompatible checkpoint: {self.model_path.name}")
                    return False
                if not bool(checkpoint["trained"]):
                    print(f"[DQN] Ignoring untrained checkpoint: {self.model_path.name}")
                    return False
                weights = [checkpoint[key] for key in sorted(
                    (key for key in checkpoint.files if key.startswith("weight_")),
                    key=lambda key: int(key.split("_")[1])
                )]
            if TF_AVAILABLE and isinstance(self.q_net, keras.Model):
                self.q_net.set_weights(weights)
            else:
                self.q_net.set_weights(weights)
            print(f"[DQN] Loaded trained {self.backend} checkpoint: {self.model_path.name}")
            return True
        except Exception as error:
            print(f"[DQN] Could not load checkpoint {self.model_path.name}: {error}")
            return False

    def optimize_layout_trajectory(self, env, max_steps: int = 25) -> List[Dict[str, Any]]:
        """Find monotonic layout improvements; use Q-values to break reward ties."""
        max_steps = max(0, min(int(max_steps), 100))
        initial_layout = env.get_layout_dict()
        trajectory = [{
            "step": 0,
            "action_desc": "Starting layout",
            "layout": initial_layout,
            "score": initial_layout["metrics"]["ergonomics_score"],
            "reward": initial_layout["metrics"]["total_reward"],
        }]
        if not env.furniture or max_steps == 0:
            return trajectory

        action_names = ["Shift right", "Shift left", "Move toward the window", "Move toward the entry", "Rotate", "Align to wall"]
        for step in range(1, max_steps + 1):
            state = env.get_state()
            q_values = self._predict(self.q_net, state[None, :])[0]
            current_reward = env.evaluate_layout()["total_reward"]
            original_furniture = copy.deepcopy(env.furniture)
            original_step_count = env.step_count
            candidates = []

            for action in range(len(original_furniture) * env.actions_per_item):
                env.furniture = copy.deepcopy(original_furniture)
                env.step_count = original_step_count
                next_state, reward, _, info = env.step(action)
                if reward > current_reward + 1e-6:
                    candidates.append((float(reward), float(q_values[action]), action,
                                       next_state, info, env.get_layout_dict()))

            env.furniture = original_furniture
            env.step_count = original_step_count
            if not candidates:
                break

            best_reward = max(candidate[0] for candidate in candidates)
            best_candidates = [candidate for candidate in candidates if abs(candidate[0] - best_reward) < 1e-6]
            _, _, action, _, info, _ = max(best_candidates, key=lambda candidate: candidate[1])
            _, _, _, _, _, layout = next(candidate for candidate in candidates if candidate[2] == action)
            item_name = info.get("modified_item", "furniture")
            action_name = action_names[info.get("action_type", 0)]
            env.load_furniture(layout["furniture"])
            env.step_count = original_step_count + 1
            trajectory.append({
                "step": step,
                "action_desc": f"{action_name} · {item_name}",
                "layout": layout,
                "score": layout["metrics"]["ergonomics_score"],
                "reward": layout["metrics"]["total_reward"],
            })

        return trajectory


class SimpleNumpyMLP:
    """One-hidden-layer MLP with real mini-batch backpropagation."""

    def __init__(self, in_dim: int, out_dim: int, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.w1 = (np.random.randn(in_dim, 64) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.w2 = (np.random.randn(64, out_dim) * np.sqrt(2.0 / 64)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        hidden = np.maximum(0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def train_on_batch(self, x: np.ndarray, targets: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        pre_activation = x @ self.w1 + self.b1
        hidden = np.maximum(0, pre_activation)
        predictions = hidden @ self.w2 + self.b2
        error = predictions - targets
        loss = float(np.mean(error ** 2))
        grad_out = (2.0 / max(1, x.shape[0] * targets.shape[1])) * error
        grad_w2 = hidden.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_hidden = (grad_out @ self.w2.T) * (pre_activation > 0)
        grad_w1 = x.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        return loss

    def get_weights(self) -> List[np.ndarray]:
        return [self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        if len(weights) != 4:
            raise ValueError("NumPy DQN checkpoint must contain four MLP weight arrays")
        expected = (self.w1.shape, self.b1.shape, self.w2.shape, self.b2.shape)
        if any(tuple(np.asarray(weight).shape) != shape for weight, shape in zip(weights, expected)):
            raise ValueError("DQN checkpoint dimensions do not match this NumPy model")
        self.w1, self.b1, self.w2, self.b2 = [np.asarray(weight, dtype=np.float32).copy() for weight in weights]

    def copy_from(self, other: "SimpleNumpyMLP") -> None:
        self.set_weights(other.get_weights())
