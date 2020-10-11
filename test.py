# Import internal modules
from agents.dqn_agent import DQNAgent

# Import external modules
import gym


if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')
    agent = DQNAgent(environment=env)

    # Test the agent
    agent.test()
