from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random


class DQNAgent:
    """

    Deep Q-learning Agent

    """
    def __init__(self, environment):
        self.env = environment
        self.state_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.episodes = 100
        self.batch_size = 32
        self.model = self._build_model()

    def _build_model(self):
        """

        Neural Net with one input layer with 4 neurons, 2 hidden layers and
        an output layer with 2 nodes since there are two possibilities (0 and 1) for Deep Q Learning

        :return: a neural network model
        """

        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))

        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))

        # Output Layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_size, activation='linear'))

        # Create the model based on the information above
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _memorize(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """

        Simply store states, actions and resulting rewards to the memory

        :param state: information we need for training the agent
        :param action: can be either 0 (left) or 1 (right)
        :param reward: Reward is 1 for every frame the pole survived
        :param next_state: information we need for training the agent
        :param done: is a boolean value telling whether the game ended or not
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def _act(self, state: np.ndarray):
        """

        :param state: information we need for training the agent
        :return: next action of the agent
        """
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return self._random_action()
        else:
            # Pick the action based on the predicted reward
            return self._greedy_action(state)

    def _greedy_action(self, state: np.ndarray):
        """

        The agent picks the action based on the predicted reward

        :param state: information we need for training the agent
        :return: next action of the agent
        """

        return np.argmax(self._get_q_table(state=state))

    def _random_action(self) -> int:
        """

        The agent acts randomly based on the action space

        :return: can be either 0 (left) or 1 (right)
        """

        return random.randrange(self.action_size)

    def _get_q_table(self, state: np.ndarray) -> np.ndarray:
        """

        Predict the reward value based on the given state

        :param state: information we need for training the agent
        :return: next action of the agent
        """

        return self.model.predict(state)

    def _replay(self, batch_size: int):
        """

        Trains the neural network with experiences in the memory.

        :param batch_size: a randomly sampled elements of the memories of size
        :return:
        """

        # Sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        # Extract informations from each memory
        for state, action, reward, next_state, done in minibatch:

            # if done, make our target reward
            target = reward

            if not done:
                # predict the future discounted reward
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])

            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the Neural Net with the state and target_f
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        """

        Start the game and train the agent given an environment.

        :return:
        """

        # Iterate the game
        for e in range(self.episodes):

            # reset state in the beginning of each game
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for time_t in range(500):
                # turn this on if you want to render
                # env.render()

                # Decide action
                action = self._act(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                # memorize the previous state, action, reward, and done
                self._memorize(state, action, reward, next_state, done)

                # make next_state given the new current state for the next frame.
                state = next_state

                # done becomes True when the game ends
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                          .format(e, self.episodes, time_t))

                    break

            # train the agent with the experience of the episode
            self._replay(batch_size=self.batch_size)

            if e % 10 == 0:
                self._save(filepath="../save/cartpole-dqn.h5")
                print("Saving trained model as {}".format("cartpole-dqn.h5"))

    def _load(self, filepath: str):
        """

        Load a pre-trained model.

        :param filepath:
        :return: a pre-trained model with the saved weights
        """
        self.model.load_weights(filepath=filepath)

    def _save(self, filepath: str):
        """

        Save after training the model.

        :param filepath:
        :return: saved weights of a pre trained model
        """

        self.model.save_weights(filepath=filepath)

    def test(self):
        """

        Load a pre-trained model and test the model in a given environment

        :return:
        """

        # Load a pre-trained model
        self._load(filepath="save/cartpole-dqn.h5")

        # Iterate the game
        for e in range(self.episodes):

            # reset state in the beginning of each game
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for time_t in range(500):
                # turn this on if you want to render
                # env.render()

                # Decide action
                action = self._act(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = self.env.step(action)

                # done becomes True when the game ends
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                          .format(e, self.episodes, time_t))

                    break
