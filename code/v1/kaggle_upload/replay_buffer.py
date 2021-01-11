import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=True,
                 q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
        print(input_shape)
        self.mem_size = max_size
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size), dtype=np.int8)
        self.terminal_memory = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_ctr = 0
        self.discrete = discrete

    def store_transition(self, state: np.ndarray, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size
        try:
            self.state_memory[index] = state
        except ValueError:
            state = state.flatten()
            state_ = state_.flatten()

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        done = done if done is not None else 0

        try:
            self.terminal_memory[index] = 1 - int(done)
            self.action_memory[index] = action
        except TypeError:
            print(f'index: {index}')
            print(f'done: {done}')
            print(f'action: {action}')
            raise IndexError()
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))
