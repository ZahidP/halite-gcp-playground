from tensorflow import keras
from typing import List, Union, Tuple, Callable


class DenseNN:

    def __init__(self, observation_shape: int, n_actions: int,
                 output_activation: Union[None, str] = None,
                 unit_scale: int = 1, learning_rate: float = 0.0001,
                 opt: str = 'Adam') -> None:
        """
        Source: https://www.kaggle.com/garethjns/deep-q-learner-starter-code

        :param observation_shape: Tuple specifying input shape.
        :param n_actions: Int specifying number of outputs
        :param output_activation: Activation function for output. Eg.
                                  None for value estimation (off-policy methods).
        :param unit_scale: Multiplier for all units in FC layers in network
                           (not used here at the moment).
        :param opt: Keras optimiser to use. Should be string.
                    This is to avoid storing TF/Keras objects here.
        :param learning_rate: Learning rate for optimiser.

        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.unit_scale = unit_scale
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.opt = opt

    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        frame_input = keras.layers.Input(name='input', shape=(self.observation_shape,))
        fc1 = keras.layers.Dense(int(self.observation_shape / 1.5 * self.unit_scale),
                                 name='fc1', activation='relu')(frame_input)
        fc2 = keras.layers.Dense(int(self.observation_shape / 3 * self.unit_scale),
                                 name='fc2', activation='relu')(fc1)
        fc3 = keras.layers.Dense(self.n_actions * 2,
                                 name='fc3', activation='relu')(fc2)
        action_output = keras.layers.Dense(units=self.n_actions, name='output',
                                           activation=self.output_activation)(fc3)

        return frame_input, action_output

    def compile(self, model_name: str = 'model',
                loss: Union[str, Callable] = 'mse') -> keras.Model:
        """
        Compile a copy of the model using the provided loss.

        :param model_name: Name of model
        :param loss: Model loss. Default 'mse'. Can be custom callable.
        """
        # Get optimiser
        if self.opt.lower() == 'adam':
            opt = keras.optimizers.Adam
        elif self.opt.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop
        else:
            raise ValueError(f"Invalid optimiser {self.opt}")

        state_input, action_output = self._model_architecture()
        model = keras.Model(inputs=[state_input], outputs=[action_output],
                            name=model_name)
        model.compile(optimizer=opt(learning_rate=self.learning_rate),
                      loss=loss)

        return model

    # def plot(self, model_name: str = 'model') -> None:
    #     keras.utils.plot_model(self.compile(model_name),
    #                            to_file=f"{model_name}.png", show_shapes=True)
    #     plt.show()
