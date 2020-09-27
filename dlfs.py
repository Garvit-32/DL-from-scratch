import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def assert_same_shape(array: ndarray, array_grad: ndarray):

    assert array.shape == array_grad.shape

    return None


class Operation(object):

    def __init__(self):

        pass

    def forward(self, input_: ndarray):

        # Stores input in the self._input instance variable
        # Calls the self._output() function.

        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        # call the self._input_grad() function

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_grad, self.input_)

        return self.input_grad

    def _output(self) -> ndarray:

        # The output method must defined for each operations

        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError()


class ParamOperation(Operation):

    # An operation with parameters

    def __init__(self, param: ndarray) -> ndarray:

        super.__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:

        # Calls self._input_grad and self._param_grad.
        # Checks appropriate shapes.

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_grad, self.input_)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        # Every subclass of ParamOperation must implement _param_grad.

        raise NotImplementedError()


class WeightMultiply(ParamOperation):

    # Weight Multiplication operation for a neural network

    def __init__(self, W: ndarray):

        super.__init__(W)

    def _output(self) -> ndarray:

        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self, B: ndarray):

        assert B.shape[0] == 1

        super.__init__(B)

    def _output(self) -> ndarray:

        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        param_grad = np.ones_like(self.param) * output_grad

        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):

    def __init__(self) -> None:

        super().__init__()

    def _output(self) -> ndarray:

        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        sigmoid_backward = self.output * (1.0 - self.output)

        input_grad = sigmoid_backward * output_grad

        return input_grad


class Linear(Operation):

    def __init__(self) -> None:

        super.__init__()

    def _output(self) -> ndarray:

        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad


class Layer(object):

    def __init__(self, neurons: int):

        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grad: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        # Must be implemented for each layer
        #
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

    def backward(self, output_grad: ndarray) -> ndarray:

        # Passes output_grad through a series of operations

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operations.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:

        # Extract the param grad for a layer operation

        self.param_grads = []
        for operation in self.operations:

            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:

        # Extract the _params from a layer's operations

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    # A fully connected layer that inherits property from "Layer"

    def __init__(self, neurons: int, activation: Operation = Sigmoid()) -> None:

        # Requires an activation function

        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:

        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.parasm.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]), BiasAdd(
            self.params[1]), self.activation]

        return None


class Loss(object):

    # The loss of a neural network

    def __init__(self):
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:

        # Compute the actual loss value

        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()
        return loss_value

    def backward(self) -> ndarray:

        # Compute the gradient of the loss w.r.t input to the loss function

        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return input_grad

    def _output(self) -> float:

        raise NotImplementedError()

    def _input_grad(self) -> ndarray:

        raise NotImplementedError()


class MeanSquaredError(Loss):

    def __init__(self):

        super().__init__()

    def _output(self) -> float:

        # Compute per observational loss

        loss = np.sum(np.power(self.prediction - self.target, 2)
                      ) / self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class NeuralNetwork(object):

    def __init__(self, layers: List(Layer), loss: Loss, seed: float = 1):

        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:

        # Passes data forward through a series of layers

        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:

        # Passes data backward through a series of layers

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:

        # Passes data forward through a layer.
        # Computes the loss.
        # Passes data backward through the layers.

        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):

        # Get the parameters for the network

        for layer in self.layers:
            yield from layer.params

    def parma_grads(self):

        # Get the gradients of the loss w.r.t parameters for the network

        for layer in self.layers:
            yield from layer.param_grads
