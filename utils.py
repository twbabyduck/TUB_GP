import numpy as np
import tensorflow as tf
from gpflow.config import default_float as floatx
from gpflow.kernels import Matern52
from gpflow_sampling.models import PathwiseGPR
from gpflow_sampling.sampling.updates import cg as cg_update

import sampler

def split(x, y):
    return x, y, x, y


def generate_data():
    kernel = Matern52(lengthscales=0.1)
    noise2 = 1e-3  # measurement noise variance

    xmin = 0.15  # range over which we observe
    xmax = 0.50  # the behavior of a function $f$
    X = tf.convert_to_tensor(np.linspace(xmin, xmax, 1024)[:, None])

    K = kernel(X, full_cov=True)
    L = tf.linalg.cholesky(tf.linalg.set_diag(K, tf.linalg.diag_part(K) + noise2))
    y = L @ tf.random.normal([len(X), 1], dtype=floatx())
    y -= tf.reduce_mean(y)  # for simplicity, center the data
    return X, y


def generate_samplers(x, y):
    return [
        sampler.Sample_Path_Sampler(x, y, [1]),
        sampler.Dummy_Sampler(x, y, [1]),
        sampler.Weight_Space_Sampler(x, y, [1]),
        sampler.Function_Space_Sampler(x, y, [1]),
        sampler.Decoupled_Sampler(x, y, [1]),
        sampler.Thompson_Sampler(x, y, [1])
    ]
