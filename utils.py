import numpy as np
import tensorflow as tf
from gpflow.config import default_float as floatx
from gpflow.kernels import Matern52
from gpflow_sampling.models import PathwiseGPR
from gpflow_sampling.sampling.updates import cg as cg_update


import sampler


def split(x, y):
    return x, y, x, y


def generate_data(nr_train_data = 4):
    kernel = Matern52(lengthscales=0.1)
    noise2 = 1e-3  # measurement noise variance

    xmin = 0.15  # range over which we observe
    xmax = 0.50  # the behavior of a function $f$
    X = tf.convert_to_tensor(np.linspace(xmin, xmax, 1024)[:, None])

    K = kernel(X, full_cov=True)
    L = tf.linalg.cholesky(tf.linalg.set_diag(K, tf.linalg.diag_part(K) + noise2))
    y = L @ tf.random.normal([len(X), 1], dtype=floatx())
    y -= tf.reduce_mean(y)  # for simplicity, center the data


    return X[0:nr_train_data], y[0:nr_train_data] #todo later: shuffle train data



def generate_samplers(x, y):
    return  [
        sampler.Weight_Space_Sampler(x, y, [1]),
        sampler.Sample_Path_Sampler(x, y, [1]),

        sampler.Function_Space_Sparse_Sampler(x,y,[1]),

        #sampler.Function_Space_Sampler(x, y, [1]),
        sampler.Dummy_Sampler(x, y, [1]),
        sampler.Decoupled_Sampler(x, y, [1]),
        sampler.Thompson_Sampler(x, y, [1])
    ]


def exponentiated_quadratic(x, x_prime, variance, lengthscale):
    squared_distance = ((x - x_prime) ** 2).sum()
    return variance * np.exp((-0.5 * squared_distance) / lengthscale ** 2)

    return K


def compute_kernel(X, X2, kernel, variance, lengthscale):
    X_s = np.asarray(X)
    X2_s = np.asarray(X2)
    # K = np.zeros((X_s.shape[0], X2_s.shape[0]))
    K = np.zeros((len(X), len(X2)))
    for i in np.arange(X_s.shape[0]):
        for j in np.arange(X2_s.shape[0]):
            K[i, j] = kernel(X_s[i], X2_s[j], variance, lengthscale)
    return K

