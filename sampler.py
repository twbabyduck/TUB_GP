import utils
import numpy as np
import time

import numpy as np
import tensorflow as tf
import GPy


from tqdm import tqdm
from itertools import count
from gpflow.kernels import Matern52
from gpflow.config import default_float as floatx
from gpflow_sampling.models import PathwiseGPR

import matplotlib.pyplot as plt

plt.rc('figure', dpi=256)
plt.rc('font', family='serif', size=12)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'''
       \usepackage{amsmath,amsfonts}
       \renewcommand{\v}[1]{\boldsymbol{#1}}''')


class Gauss_Process_Sampler():

    def __init__(self, x, y):
        self.train_x, self.train_y, self.test_x, self.test_y = utils.split(x, y)

    def sample_from_prior(self, nr_observations):
        pass

    def select_hyperparams(self, hyperparams):
        if hyperparams == None:
            self.hyperparams = random_hyperparams()
        else:
            self.hyperparams = hyperparams

    def random_hyperparams(self):
        pass

    def fit(self):
        pass

    def sample_from_posterior(self, X_test, y_test):
        pass

    def reset(self):
        pass


    def simple_plot(self, Xnew, fnew, x_train, y_train):
        for iteration in np.arange(fnew.shape[0]):
            plt.plot(Xnew,fnew[iteration],zorder=1) #c=np.zeros(Xnew.shape[0])+iteration
        plt.scatter(x_train, y_train,color='black',zorder=2)
        plt.show()


    def plot(self, Xnew, xmins, fmins, fnew, mu, sigma2, lower, upper):
        """

        :param Xnew:  1024 locations on x axis (lowest value: 0, heightest value: 1)
        :param xmins: ?
        :param fmins: ?
        :param fnew:
        :param mu:
        :param sigma2:
        :param lower:
        :param upper:
        :return:
        """
        fig, ax = plt.subplots(figsize=(7, 3))

        # Indicate where the training data is located
        for x in self.train_x:
            ax.axvline(x, linewidth=4, zorder=0, alpha=0.25, color='silver')

        # Show gold-standard quantiles
        ax.plot(Xnew, mu, '--k', linewidth=1.0, alpha=0.8)
        ax.plot(Xnew, mu + tf.math.ndtri(lower) * tf.sqrt(sigma2), '--k', linewidth=0.75, alpha=0.8)
        ax.plot(Xnew, mu + tf.math.ndtri(upper) * tf.sqrt(sigma2), '--k', linewidth=0.75, alpha=0.8)

        cmap = plt.get_cmap('tab10')
        colors = cmap(range(self.model.paths.sample_shape[0]))

        for i, xmin, fmin, f in zip(count(), xmins, fmins, fnew):
            print("++")
            print(i)
            print("++")
            ax.plot(Xnew, f, zorder=99, color=colors[i], alpha=2 / 3, linewidth=1.0)
            # ax.scatter(xmin, fmin, zorder=999, color=colors[i], alpha=0.9, linewidth=2 / 3, marker='o', s=16, edgecolor='k')


        _ = ax.set_ylabel(r'$(f \mid \v{y})(\cdot)$')
        _ = ax.set_xlim(0, 1)
        _ = ax.set_xlabel(r'$\v{x} \in \mathbb{R}$')
        plt.savefig('plots/plot' + str(time.time()) + '.png')

    def wasserstein_distance(self):
        return 0.01


class Dummy_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y):
        super().__init__(x, y)

    def sample_from_prior(self, nr_observations):
        return np.ones(nr_observations)

    def fit(self):
        pass


class Sample_Path_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.setup()

    def setup(self):
        self.kernel = Matern52(lengthscales=0.1)
        self.noise2 = 1e-3  # measurement noise variance

        self.model = PathwiseGPR(data=(self.train_x, self.train_y), kernel=self.kernel, noise_variance=self.noise2)
        self.paths = self.model.generate_paths(num_samples=4, num_bases=1024)
        _ = self.model.set_paths(self.paths)  # use a persistent set of sample paths

    def fit(self):
        Xinit = tf.random.uniform(self.paths.sample_shape + [32, 1], dtype=floatx())
        self.Xvars = tf.Variable(Xinit, constraint=lambda x: tf.clip_by_value(x, 0, 1))

        @tf.function
        def closure(sample_axis=0):
            """
            Passing sample_axis=0 indicates that the 0-th axis of Xnew
            should be evaluated 1-to-1 with the individuals paths.
            """
            return self.model.predict_f_samples(Xnew=self.Xvars, sample_axis=sample_axis)

        optimizer = tf.keras.optimizers.Adam()
        for step in tqdm(range(10)):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.Xvars)
                fvals = closure()

            grads = tape.gradient(fvals, self.Xvars)
            optimizer.apply_gradients([(grads, self.Xvars)])



    def plot(self):
        def take_along_axis(arr: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
            """
            Tensorflow equivalent of <numpy.take_along_axis>
            """
            _arr = tf.convert_to_tensor(arr)
            _idx = tf.convert_to_tensor(indices)
            _axis = arr.shape.ndims + axis if (axis < 0) else axis

            components = []
            for i, (size_a, size_i) in enumerate(zip(_arr.shape, _idx.shape)):
                if i == _axis:
                    components.append(tf.range(size_i, dtype=_idx.dtype))
                elif size_a == 1:
                    components.append(tf.zeros(size_i, dtype=_idx.dtype))
                else:
                    assert size_i in (1, size_a), \
                        ValueError(f'Shape mismatch: {_arr.shape} vs {_idx.shape}')
                    components.append(tf.range(size_a, dtype=_idx.dtype))

            mesh = tf.meshgrid(*components, indexing='ij')
            mesh[_axis] = tf.broadcast_to(_idx, mesh[0].shape)
            indices_nd = tf.stack(mesh, axis=-1)
            return tf.gather_nd(arr, indices_nd)

        lower = tf.cast(0.025, floatx())
        upper = tf.cast(0.975, floatx())

        Xnew = np.linspace(0, 1, 1024)[:, None]  # abstract this ->x_values
        fnew = tf.squeeze(self.model.predict_f_samples(Xnew))  # abstract this -> array of y_values
        mu, sigma2 = map(tf.squeeze, self.model.predict_f(Xnew))  # abstract - mean, quantiles

        # Plot samples paths and pathwise minimizers
        fcand = self.model.predict_f_samples(self.Xvars, sample_axis=0)
        index = tf.argmin(fcand, axis=1)
        xmins = take_along_axis(self.Xvars, index[..., None], axis=1)
        fmins = take_along_axis(fcand, index[..., None], axis=1)
        super().plot(Xnew, xmins, fmins, fnew, mu, sigma2, lower, upper)


class Function_Space_Sparse_Sampler():

    def __init__(self, X_train, y_train, nr_inducing_points):

        self.variance = 1
        self.lengthscale = 10
        optimize_inducing = False

        self.X = np.asarray(X_train).reshape(-1, 1)
        self.y = np.asarray(y_train).reshape(-1, 1)
        self.u = nr_inducing_points
        self.Z = np.zeros(6)
        for i in range(6):
            self.Z[i] = np.random.normal(np.mean(X_train), 1)
        self.Z = self.Z.reshape(-1, 1)
        self.kernel = GPy.kern.RBF(input_dim=1, variance=self.variance, lengthscale=self.lengthscale)
        self.m = GPy.models.SparseGPRegression(self.X, self.y, Z=self.Z, kernel=self.kernel)
        # m.plot()

    def optimize_model(self):
        self.m.inducing_inputs.fix()
        self.m.optimize(messages=True)
        self.m.optimize_restarts(num_restarts=10)

    def optimize_inducing_locations(self):
        self.m.randomize()
        self.m.Z.unconstrain()
        self.m.optimize('bfgs')



    def sample_from_posterior(self, X_test, y_test):

        if self.optimize_inducing_locations():
            self.optimize_inducing_locations()

        reshaped_Test = np.asarray(X_test).reshape(-1, 1)
        posteriorTestY = self.m.posterior_samples_f(reshaped_Test, full_cov=True, size=1)

        simY, simMse = self.m.predict(reshaped_Test)
        x_predictions = np.linspace(1, len(X_test), len(X_test))
        Y_predictions = posteriorTestY.reshape(len(posteriorTestY))
        plt.plot(x_predictions, X_test, 'rx')
        plt.plot(x_predictions, Y_predictions, 'b-')
        plt.show()

        # return posteriorTestY

    def sample_from_prior(self, nr_samples):
        ready_for_kernel = np.vstack((self.Z.flatten(), self.Z.flatten())).T
        K = k.K(ready_for_kernel)
        x_predictions = np.linspace(1, len(self.Z), len(self.Z))
        for i in range(nr_samples):
            y_sample = np.random.multivariate_normal(mean=np.zeros(self.Z.size), cov=K)
            # print(y_sample)
            plt.plot(x_predictions.flatten(), y_sample.flatten())


class Function_Space_Exact(): #todo: check if this is really Function Space or something else

    def __init__(self, X_train, y_train):
        self.variance = 1
        self.lengthscale = 10
        self.X = np.asarray(X_train).reshape(-1, 1)

        self.y = np.asarray(y_train).reshape(-1, 1)
        # self.kernel = kernel
        # self.variance = variance
        # self.lengthscale = lengthscale
        self.kernel = GPy.kern.RBF(input_dim=1, variance=self.variance, lengthscale=self.lengthscale)
        self.m = GPy.models.GPRegression(self.X, self.y, self.kernel)
        # self.update_inverse()

    def optimize_model(self):
        self.m.optimize(messages=True)
        m.optimize_restarts(num_restarts=10)

    def sample_from_posterior(self, X_test, y_test):
        reshaped_Test = np.asarray(X_test).reshape(-1, 1)
        posteriorTestY = self.m.posterior_samples_f(reshaped_Test, full_cov=True, size=1)
        simY, simMse = self.m.predict(reshaped_Test)
        x_predictions = np.linspace(1, len(X_test), len(X_test))
        Y_predictions = posteriorTestY.reshape(len(posteriorTestY))
        plt.plot(x_predictions, X_test, 'rx')
        plt.plot(x_predictions, Y_predictions, 'b-')
        plt.show()
        # return posteriorTestY

    def sample_from_prior(self, nr_samples):
        ready_for_kernel = np.vstack((self.X.flatten(), self.X.flatten())).T
        K = k.K(ready_for_kernel)
        x_predictions = np.linspace(1, len(self.X), len(self.X))
        for i in range(nr_samples):
            y_sample = np.random.multivariate_normal(mean=np.zeros(self.X.size), cov=K)
            # print(y_sample)
            plt.plot(x_predictions.flatten(), y_sample.flatten())




class Function_Space_Sampler():

    def __init__(self, X_train, y_train, sigma2, kernel, variance, lengthscale):
        self.X = X_train
        self.y = y_train
        self.sigma2 = sigma2
        self.kernel = kernel
        self.variance = variance
        self.lengthscale = lengthscale
        self.K = utils.compute_kernel(self.X, self.X, kernel, variance, lengthscale)
        self.update_inverse()

    def update_inverse(self):
        # Preompute the inverse covariance and some quantities of interest
        ## NOTE: This is not the correct *numerical* way to compute this! It is for ease of use.
        self.Kinv = np.linalg.inv(self.K + self.sigma2 * np.eye(self.K.shape[0]))
        # the log determinant of the covariance matrix.
        self.logdetK = np.linalg.det(self.K + self.sigma2 * np.eye(self.K.shape[0]))
        # The matrix inner product of the inverse covariance
        self.Kinvy = np.dot(self.Kinv, self.y)
        self.yKinvy = (self.y * self.Kinvy).sum()

   # def log_likelihood(self):
        # use the pre-computes to return the likelihood
    #    return -0.5 * (self.K.shape[0] * np.log(2 * np.pi) + self.logdetK + self.yKinvy)

  #  def objective(self):
        # use the pre-computes to return the objective function
  #      return -self.log_likelihood()

    def sample_from_posterior(self, X_test, y_test):
        K_star = utils.compute_kernel(self.X, X_test, self.kernel, self.variance, self.lengthscale)
        K_starstar = utils.compute_kernel(X_test, X_test, self.kernel, self.variance, self.lengthscale)
        A = np.dot(self.Kinv, K_star)
        mu_f = np.dot(A.T, self.y)
        C_f = K_starstar - np.dot(A.T, K_star)
        x_predictions = np.linspace(1,len(X_test),len(X_test))
        plt.plot(x_predictions, x1_test, 'rx')
        plt.plot(x_predictions, mu_f, 'b-')
        #return mu_f, C_f

    def sample_from_prior(self, nr_samples):
        K = utils.compute_kernel(self.X, self.X, utils.exponentiated_quadratic, self.variance, self.lengthscale)
        x_predictions = np.linspace(1,len(self.X),len(self.X))

        for i in range(nr_samples):
            y_sample = np.random.multivariate_normal(mean=np.zeros(self.X.size), cov=K)
            # print(y_sample)
            plt.plot(x_predictions.flatten(), y_sample.flatten())

class Weight_Space_Sampler(Gauss_Process_Sampler):

    def __init__(self, x, y, l):
        super().__init__(x, y)
        self.L = l
        self.number_functions = 4

    def f(self, x, l, weights=None):
        sum=0
        for _l in np.arange(l) + 1:
            if weights is None:
                weights = np.random.normal(0, 1)
            _phi = self.phi(x, l)
            sum+=np.dot(weights[_l-1], _phi)
        return sum

    def phi(self, x, _l):
        ri = np.random.uniform(0, np.pi)
        theta = np.zeros(x.shape[0]) + 1  # todo
        return np.sqrt((2 / _l)) * np.cos(np.dot(x, theta) + ri)

    def PHI(self, X):
        data = np.zeros((X.shape[0], self.L), dtype=np.float64)
        for index_l, _l in enumerate(np.arange(self.L) + 1):
            for index_x, x in enumerate(X):
                data[index_x, index_l] = self.phi(x, _l)
        return data

    def fit(self):

        _PHI = self.PHI(self.train_x)

        variance = 1  # todo variance to identity - which variance?
        inverse = np.linalg.inv((np.dot(_PHI.T, _PHI) + (np.identity(self.L) * (variance ** 2))))
        self.post_w_mean = np.dot(np.dot(inverse, _PHI.T), self.train_y)
        self.post_w_cov = inverse * (variance ** 2)
        self.weights = np.random.multivariate_normal( np.squeeze(self.post_w_mean), self.post_w_cov)

    def sample_from_posterior(self, X_test, y_test):
        x_points = 1000
        self.fit()
        self.fnew = np.zeros((self.number_functions, x_points))
        for f_iteration in np.arange(self.number_functions):

            # create data for this function
            self.Xnew = np.linspace(0, 1, x_points)[:, None]  # abstract this ->x_values
            y_values = []
            for x_index, x in enumerate(self.Xnew):
                self.fnew[f_iteration, x_index] = self.f(x, self.L, self.weights)
        self.plot()

    def plot(self):
        super().simple_plot(self.Xnew.reshape(self.Xnew.shape[0]), self.fnew, self.train_x, self.train_y)
        '''
        normal plot seems to be to complicated at the moment -> check later
        xmins = np.array([0,0,0,0])
        fmins = np.array([0,0,0,0])
        mu = np.zeros(1024) +  0.074665
        sigma2 = np.zeros(1024) +  0.07
        lower = 0.01
        upper=0.9
        #super().plot(self.Xnew, xmins, fmins, self.fnew, mu, sigma2, lower, upper)
        '''


class Decoupled_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y):
        super().__init__(x, y)

    def fit(self):
        pass


class Thompson_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y):
        super().__init__(x, y)

    def fit(self):
        pass


class Low_Level_Weight_Space_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y):
        super().__init__(x, y)

    def fit(self):
        pass
