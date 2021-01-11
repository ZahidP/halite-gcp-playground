from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

def ActorModel(num_actions, in_):
    common = tf.keras.layers.Dense(128, activation='tanh')(in_)
    common = tf.keras.layers.Dense(32, activation='tanh')(common)
    common = tf.keras.layers.Dense(num_actions, activation='softmax')(common)

    return common


def CriticModel(in_):
    common = tf.keras.layers.Dense(128)(in_)
    common = tf.keras.layers.ReLU()(common)
    common = tf.keras.layers.Dense(32)(common)
    common = tf.keras.layers.ReLU()(common)
    common = tf.keras.layers.Dense(1)(common)

    return common

input_ = tf.keras.layers.Input(shape=[441,])
model = tf.keras.Model(inputs=input_, outputs=[ActorModel(5,input_),CriticModel(input_)])
model.summary()


class Agent(object):

    def __init__(
            self,
            lr1,
            lr2,
            gamma,
            n_actions,
            input_dims,
            fc1_dims,
            fc2_dims
    ):
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims

    def build_actor_critic_network(self):

        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        # ...