import numpy as np
import tensorflow as tf
from gpflow.config import default_float as floatx
from gpflow.kernels import Matern52
from gpflow_sampling.models import PathwiseGPR
from gpflow_sampling.sampling.updates import cg as cg_update
import GPy

import sampler
import pods

def split(x, y):
    return x, y, x, y


def generate_fourier_basis_data(nr_data):
    x=np.linspace(0,10,nr_data)[::-1]
    y=list(map(lambda _x: np.cos(_x), x))[::-1]
    #y = np.ones(nr_data)
    return np.array(list(map(lambda _x: np.array([_x]), x))),y

def generate_data(nr_train_data=4):
    data = pods.datasets.olympic_marathon_men()
    return data['X'][0:nr_train_data], data['Y'][0:nr_train_data]


def generate_data5(nr_data):
    data_x = np.random.uniform(0,1,nr_data)
    data_y = np.random.normal(0,1,nr_data)
    return data_x,data_y

def generate_data4(nr_train_data, shuffle):
    N = 50
    noise_var = 0.05

    X = np.linspace(0, 10, 50)[:, None]
    print(X.shape)
    k = GPy.kern.RBF(1)
    y = np.random.multivariate_normal(np.zeros(N), k.K(X) + np.eye(N) * np.sqrt(noise_var)).reshape(-1, 1)
    #if shuffle:
        #X, y = shuffle(X, y)
    return X[0:nr_train_data], y[0:nr_train_data]

def generate_data3(nr_train_data=4):
    import pandas as pd
    data = pd.read_csv("drinks.csv")
    data = pd.DataFrame(data)

    x1 = data['beer_servings']
    x1_train = data['beer_servings'][1:101]
    x1_test = data['beer_servings'][101:194]
    x2_train = data['spirit_servings'][1:101]
    x2_test = data['spirit_servings'][101:194]
    x3_train = data['wine_servings'][1:101]
    x3_test = data['wine_servings'][101:194]
    x4_train = data['total_litres_of_pure_alcohol'][1:101]
    x4_test = data['total_litres_of_pure_alcohol'][101:194]
    x_pred_train = np.linspace(1, 100, 100)
    x_pred_test = np.linspace(1, 92, 92)
    x1_test = x1_test - np.mean(x1_train)
    x1_train = x1_train - np.mean(x1_train)

    return x2_train[0:nr_train_data], x1_train[0:nr_train_data], x2_test[0:nr_train_data] , x1_test[0:nr_train_data]

def _generate_data(nr_train_data = 4):
    return generate_data_2(nr_train_data)
    kernel = Matern52(lengthscales=0.1)
    noise2 = 1e-3  # measurement noise variance


    xmin = 0.15  # range over which we observe
    xmax = 0.50  # the behavior of a function $f$
    X = tf.convert_to_tensor(np.linspace(xmin, xmax, 1024)[:, None])

    K = kernel(X, full_cov=True)
    L = tf.linalg.cholesky(tf.linalg.set_diag(K, tf.linalg.diag_part(K) + noise2))
    y = L @ tf.random.normal([len(X), 1], dtype=floatx())
    y -= tf.reduce_mean(y)  # for simplicity, center the data

    import pdb;
    pdb.set_trace()


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

