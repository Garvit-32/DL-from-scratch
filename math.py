import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray
from typing import List, Callable, Dict


def square(x):
    return np.power(x, 2)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


# 'A function to calculate deriv'


def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, diff: float = 0.001) -> ndarray:
    return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)


# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of function
Chain = List[Array_Function]


def chain_length_2(chain: Chain, x: ndarray) -> ndarray:
    # 'Evaluate two function in a row in a chain'

    assert len(chain) == 2

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))


def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    # Uses chain rule (f2(f1(x))' = f2'(f1(x)) * f1'(x)

    assert len(chain) == 2

    assert input_range.ndim == 1

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    df1dx = deriv(f1, input_range)

    # df2/dx
    df2du = deriv(f2, f1(input_range))

    return df1dx * df2du


def plot_chain(ax, chain: Chain, input_range: ndarray, length: int = 2) -> ndarray:

    assert input_range.ndim == 1

    if length == 2:
        output_range = chain_length_2(chain, input_range)
    elif length == 3:
        output_range = chain_length_3(chain, input_range)
    ax.plot(input_range, output_range)


def plot_chain_deriv(ax, chain: Chain, input_range: ndarray, length: int = 2) -> ndarray:

    if length == 2:
        output_range = chain_deriv_2(chain, input_range)
    elif length == 3:
        output_range = chain_deriv_3(chain, input_range)
    ax.plot(input_range, output_range)


def chain_length_3(chain: Chain, x: ndarray) -> ndarray:

    # Evaluates three functions in a row, in a "Chain".
    assert len(chain) == 3

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))


def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:

    # Uses the chain rule to compute the derivative of three nested functions:
    # (f3(f2(f1)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)

    assert len(chain) == 3

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_of_x = f1(input_range)

    # f2(f1(x))
    f2_of_x = f2(f1_of_x)

    # df3du
    df3du = deriv(f3, f2_of_x)

    # df2du
    df2du = deriv(f2, f1_of_x)

    # df1dx
    df1dx = deriv(f1, input_range)

    return df1dx * df2du * df3du


def multiple_inputs_add(x: ndarray, y: ndarray, sigma: Array_Function) -> float:

    # forward pass

    assert x.shape == y.shape

    a = x + y
    return sigma(a)


def multiple_inputs_add_backward(x: ndarray, y: ndarray, sigma: Array_Function) -> float:

    # Computes the derivative of this simple function with respect to both inputs

    a = x + y

    dsda = deriv(sigma, a)

    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:

    # Computes the forward pass of a matrix multiplication

    assert X.shape[1] == W.shape[0]

    return np.dot(X, W)


def matmul_backward_first(X: ndarray, W: ndarray) -> ndarray:

    # X =[x1,x2,x3,x4,x5]
    # w= [w1
    #     w2
    #     w3]
    # N = X.W =  [x1 * w1 , x2 * w2 ,x3 * w3]
    # And looking at this, we can see that if
    # for example, x1 changes by ϵ units, then N will change by w1 × ϵ units

    # dN/dX = [w1,w2,w3,w4,w5]

    # dN/dX = transpose of W

    dNdX = np.transpose(W, (1, 0))
    return dNdX


def matrix_forward_extra(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:

    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)

    S = sigma(N)

    return S


def matrix_function_backward_1(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:

    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)

    S = sigma(N)

    dSdN = deriv(sigma, N)

    dNdX = np.transpose(W, (1, 0))

    return np.dot(dSdN, dNdX)


def matrix_function_forward_sum(X: ndarray, W: ndarray, sigma: Array_Function) -> float:

    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)

    S = sigma(N)

    L = np.sum(S)

    return L


def matrix_function_backward_sum_1(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:

    # Compute derivative of matrix function with a sum with respect to the first matrix input

    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)

    S = sigma(N)

    L = np.sum(S)

    dLdS = np.ones_like(S)

    dSdN = deriv(sigma, N)

    dLdN = dLdS * dSdN

    dNdX = np.transpose(W, (1, 0))

    dLdX = np.dot(dSdN, dNdX)

    return dLdX
