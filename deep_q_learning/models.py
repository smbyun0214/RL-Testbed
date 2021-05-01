from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.activations import relu


def create_dqn_model(num_actions):
    # The input to the neural network consists of an 84 x 84 x 4 image
    # produced by the preprocessing map φ.
    inputs = Input(shape=(84, 84, 4))
    # The first hidden layer convolves 32 filters of 8 x 8 with stride 4,
    # with the input image and applies a rectifier nonlinearity.
    hidden1 = Conv2D(
        filters=32,
        kernel_size=(8, 8),
        strides=4,
        activation=relu
    )(inputs)
    # The second hidden layer convolves 64 filters of 4 x 4 with stride 2,
    # again followed by a rectifier nonlinearity.
    hidden2 = Conv2D(
        filters=64,
        kernel_size=(4, 4),
        strides=2,
        activation=relu
    )(hidden1)
    # This is followed by a third convolutional layer that
    # convolves 64 filters of 3 x 3, with stride 1 followed by a rectifier. 
    hidden3 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        activation=relu
    )(hidden2)
    flatten = Flatten()(hidden3)
    # The final hidden layer is fully-connected and consists of 512 rectifier units.
    hidden4 = Dense(units=512, activation=relu)(flatten)
    # The output layer is a fully-connected linear layer
    # with a single output for each valid action.
    outputs = Dense(units=num_actions)(hidden4)

    return Model(inputs=inputs, outputs=outputs)


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.activations import relu


def create_drqn_model(num_actions, trainable=True):
    # The input to the neural network consists of an 84 x 84 x 4 image produced by the preprocessing map φ.
    inputs = Input(shape=(10, 84, 84, 1))
    # The first hidden layer convolves 32 filters of 8 x 8 with stride 4,
    # with the input image and applies a rectifier nonlinearity.
    hidden1 = TimeDistributed(Conv2D(
        filters=32,
        kernel_size=(8, 8),
        strides=4,
        activation=relu
    ))(inputs)
    # The second hidden layer convolves 64 filters of 4 x 4 with stride 2,
    # again followed by a rectifier nonlinearity.
    hidden2 = TimeDistributed(Conv2D(
        filters=64,
        kernel_size=(4, 4),
        strides=2,
        activation=relu
    ))(hidden1)
    # This is followed by a third convolutional layer that convolves 64 filters of 3 x 3,
    # with stride 1 followed by a rectifier. 
    hidden3 = TimeDistributed(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        activation=relu
    ))(hidden2)

    flatten = TimeDistributed(Flatten())(hidden3)
    # replacing only its first fully connected layer with a recurrent LSTM layer of the same size.
    hidden4 = LSTM(units=512)(flatten)
    # The output layer is a fully-connected linear layer with a single output for each valid action.
    outputs = Dense(units=num_actions)(hidden4)

    return Model(inputs=inputs, outputs=outputs, trainable=trainable)