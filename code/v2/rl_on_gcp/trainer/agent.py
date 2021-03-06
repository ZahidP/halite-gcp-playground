import numpy as np  # linear algebra
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Conv1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from trainer.replay_buffer import ReplayBuffer
from trainer.dense_nn import DenseNN
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Observation


def build_conv_dqn(lr, n_actions, input_dims, fc1_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                     input_shape=(input_dims,None), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                     data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                     data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    return model


def build_dense_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims, fc3_dims=8):
    model = keras.Sequential(
        [keras.layers.Dense(fc1_dims, input_shape=(input_dims,)),
         keras.layers.Activation('relu'),
         keras.layers.Dense(fc2_dims),
         keras.layers.Activation('relu'),
         keras.layers.Dense(fc3_dims),
         keras.layers.Activation('relu'),
         keras.layers.Dense(n_actions)]
    )

    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')

    return model


class Agent(object):

    def __init__(self,
                 alpha,
                 gamma,
                 n_actions,
                 epsilon,
                 batch_size,
                 input_dims,
                 epsilon_dec=0.985,
                 epsilon_end=0.05,
                 win_reward=5,
                 replace=20,
                 fc1_dims=16,
                 fc2_dims=16,
                 mem_size=100_000,
                 fname='dqn_model.h5',
                 verbose=False,
                 agent_type='default',
                 nnet_type='dense'
                 ):
        """
        gamma: discount factor
        epsilon: how often we choose the random action

        """
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.model_file = fname
        self.win_reward = win_reward
        self.replace = replace
        self.learn_step = 0
        # self.state_converter = state_converter
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.verbose = verbose
        self.chose_random = False
        self.agent_type = agent_type

        if nnet_type == 'dense':
            dense_nn = DenseNN(
                n_actions=n_actions,
                unit_scale=2,
                observation_shape=input_dims
            )
            self.q_online = dense_nn.compile('q_online', 'mse')
            self.q_offline = dense_nn.compile('q_offline', 'mse')
        else:
            self.q_online = build_conv_dqn(alpha, fc1_dims=fc1_dims, input_dims=input_dims, n_actions=n_actions)
            self.q_offline = build_conv_dqn(alpha, fc1_dims=fc1_dims, input_dims=input_dims, n_actions=n_actions)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        return reward

    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_offline.set_weights(self.q_online.get_weights())

    def get_action(
            self,
            state,
            game,
            step,
            verbose=True
    ):
        """
        Given a particular state, select the 2 highest values actions, or 2 random actions.

        :param state:
        :param game: Which game number this is (used for
        :return:
        """

        rand = np.random.rand()
        if rand < self.epsilon:
            top_action_index = np.random.choice(self.action_space, 1)[0]
            # see here!
            actions = np.zeros(len(self.action_space))
            actions[top_action_index] = 1
            self.chose_random = True
        else:
            state = np.array([state])
            # pass the state through the network
            # and select the best action

            pred = self.q_online.predict(state)
            action_values = pred[0]

            top_action_index = np.argmax(action_values)
            actions = np.zeros(len(self.action_space))

            # try:
            actions[top_action_index] = 1

            self.chose_random = False

            if verbose and ((game % 10) == 0) and (step % 10 == 0):
                print(f'Game: {game}, Step: {step}')
                print('action values')
                print(action_values)
                print('top_action_index')
                print(top_action_index)
                print('actions')
                print(actions)

        return top_action_index

    def learn(self, step_num, episode_num):
        verbose = self.verbose

        # print(f'Learning with agent type: {self.agent_type}')

        verbose = verbose and (self.memory.mem_ctr == self.batch_size)

        # this is a temporal difference learning method --> we learn on each step
        # when we start, do we start with random or all zeros?
        if self.memory.mem_ctr < self.batch_size:
            return

        self.learn_step += 1

        self.replace_target_network()

        if verbose and ((episode_num % 10) == 0) and (step_num % 10 == 0):
            print('\n ================')
            print('learning - game: {}, iteration: {}'.format(episode_num, step_num))
            print('Mean reward: {}'.format(np.mean(self.memory.reward_memory)))

        # Here we sample non-sequential memory. We don't want to sample sequential
        # memory because this results in correlation (23:45 in video)
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        if verbose:
            print('states: {}, actions: {}, rewards: {}'.format(len(state), len(action), len(reward)))

        # feed set of states through the model

        if verbose and ((episode_num % 10) == 0) and (step_num % 10 == 0):
            print('Predicting q_target with input {}'.format(state.shape))
        q_network = self.q_offline.predict(state)
        if verbose:
            print('Predicting q_next with {}'.format(state.shape))
        q_next = self.q_offline.predict(new_state)

        q_network = q_network.copy()

        # this is a point of contention
        # TODO: return to this (22:00 in video)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_network[batch_index, action] = reward + \
                                             self.gamma * np.max(q_next, axis=1) * (1 - done)

        if verbose and ((episode_num % 10) == 0) and (step_num % 10 == 0):
            print('Updated q_target with shape: {}'.format(q_network.shape))

        self.q_online.train_on_batch(state, q_network)

        if verbose and ((episode_num % 10) == 0) and (step_num % 10 == 0):
            print('Training complete')

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end \
            else self.epsilon

    def save_weights(self, model_path):
        if not model_path:
            model_path = self.model_file
        self.q_online.save_weights(model_path)

    def load_weights(self, model_path):
        if not model_path:
            model_path = self.model_file
        self.q_online.load_weights(model_path)
