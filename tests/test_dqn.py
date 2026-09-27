"""
Unit Tests for SmartSpace AI Interior Environment & DQN Agent
"""
import os
import unittest
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import SAMPLE_ROOMS
from engine.interior_env import InteriorEnv
from engine.dqn_agent import DQNAgent

class TestDQNEnvironment(unittest.TestCase):
    def setUp(self):
        self.room_config = SAMPLE_ROOMS["master_bedroom"]
        self.env = InteriorEnv(self.room_config)

    def test_state_shape(self):
        state = self.env.reset()
        self.assertEqual(len(state), self.env.state_dim)

    def test_step_function(self):
        self.env.reset()
        next_state, reward, done, info = self.env.step(0)
        self.assertEqual(len(next_state), self.env.state_dim)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_trajectory_optimization(self):
        agent = DQNAgent(state_dim=self.env.state_dim, action_dim=self.env.action_dim)
        trajectory = agent.optimize_layout_trajectory(self.env, max_steps=4)
        self.assertGreater(len(trajectory), 0)
        self.assertIn("score", trajectory[-1])

if __name__ == '__main__':
    unittest.main()
