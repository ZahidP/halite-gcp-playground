from tensorflow import keras
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