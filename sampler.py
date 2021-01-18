import utils
import numpy as np

import numpy as np
import tensorflow as tf

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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

class Gauss_Process_Sampler():

    def __init__(self, x, y, test_point_indices):
        self.test_point_indices = test_point_indices
        self.train_x, self.train_y, self.test_x, self.test_y =  utils.split(x,y)

    def sample_from_prior(self, nr_observations):
        pass

    def select_hyperparams(self,hyperparams):
        if hyperparams==None:
            self.hyperparams = random_hyperparams()
        else:
            self.hyperparams = hyperparams

    def random_hyperparams(self):
        pass

    def fit(self):
        pass

    def sample_from_posterior(self):
        pass

    def reset(self):
        pass

    def plot(self, Xnew, xmins, fmins, fnew,mu,sigma2,lower,upper):
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
            #ax.scatter(xmin, fmin, zorder=999, color=colors[i], alpha=0.9, linewidth=2 / 3, marker='o', s=16, edgecolor='k')


        _ = ax.set_ylabel(r'$(f \mid \v{y})(\cdot)$')
        _ = ax.set_xlim(0, 1)
        _ = ax.set_xlabel(r'$\v{x} \in \mathbb{R}$')
        plt.savefig('plots/plot.png')

    def wasserstein_distance(self):
        return 0.01



class Dummy_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y, test_point_indices):
        super().__init__(x, y, test_point_indices)

    def sample_from_prior(self, nr_observations):
        return np.ones(nr_observations)

    def fit(self):
        pass


class Sample_Path_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y, test_point_indices):
        super().__init__(x, y, test_point_indices)
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


        Xnew = np.linspace(0, 1, 1024)[:, None] #abstract this ->x_values
        fnew = tf.squeeze(self.model.predict_f_samples(Xnew)) #abstract this -> array of y_values
        mu, sigma2 = map(tf.squeeze, self.model.predict_f(Xnew)) #abstract - mean, quantiles


        # Plot samples paths and pathwise minimizers
        fcand = self.model.predict_f_samples(self.Xvars, sample_axis=0)
        index = tf.argmin(fcand, axis=1)
        xmins = take_along_axis(self.Xvars, index[..., None], axis=1)
        fmins = take_along_axis(fcand, index[..., None], axis=1)
        super().plot(Xnew, xmins, fmins,fnew,mu,sigma2,lower,upper)

class Weight_Space_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y, test_point_indices):
        super().__init__(x, y, test_point_indices)

class Function_Space_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y, test_point_indices):
        super().__init__(x, y, test_point_indices)

    def fit(self):
        pass

class Decoupled_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y, test_point_indices):
        super().__init__(x, y, test_point_indices)

    def fit(self):
        pass


class Thompson_Sampler(Gauss_Process_Sampler):
    # gdmarmerola
    def __init__(self, n_random_samples, objective, x_bounds, interval_resolution=1000):
                
        # the number of random samples before starting the optimization
        self.n_random_samples = n_random_samples
        
        # the objective function is what we are trying to optimize
        self.objective = objective
        
        # the bounds specify the interval of x we can work
        self.bounds = x_bounds
        
        # interval resolution 
        # defined how many points to represent the posterior sample
        self.interval_resolution = interval_resolution
        self.X_grid = np.linspace(self.bounds[0], self.bounds[1], self.interval_resolution)
        
        # initialize design matrix
        self.X = np.array([])
        # initialize target variable
        self.y = np.array([])
        
    def fit(self, x, y):
        # specify Matern Kernel
        K = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
        gp = GaussianProcessRegressor(kernel=K)
        return gp.fit(x,y)
    
    # process of choosing next point
    def choose_next_sample(self):
        
        # if we do not have enough samples, sample randomly from bounds
        if self.X.shape[0] < self.n_random_samples:
            next_sample = np.random.uniform(self.bounds[0], self.bounds[1],1)[0]
        
        # if we do, we fit the GP and choose the next point based on the posterior draw minimum
        else:
            # 1. Fit the GP to the observations we have
            self.gp = self.fit(self.X.reshape(-1,1), self.y)
            
            # 2. Draw one sample (a function) from the posterior
            posterior_sample = self.gp.sample_y(self.X_grid.reshape(-1,1), 1).T[0]
            
            # 3. Choose next point as the optimum of the sample
            which_min = np.argmin(posterior_sample)
            next_sample = self.X_grid[which_min]
        
            # let us also get the std from the posterior, for visualization purposes
            posterior_mean, posterior_std = self.gp.predict(self.X_grid.reshape(-1,1), return_std=True)
        
        # let us observe the objective and append this new data to our X and y
        next_observation = self.objective(next_sample)
        self.X = np.append(self.X, next_sample)
        self.y = np.append(self.y, next_observation)
        
        # return everything if possible
        try:
            # returning values of interest
            return self.X, self.y, self.X_grid, posterior_sample, posterior_mean, posterior_std
        
        except:
            return (self.X, self.y, self.X_grid, np.array([np.mean(self.y)]*self.interval_resolution), 
                    np.array([np.mean(self.y)]*self.interval_resolution), np.array([0]*self.interval_resolution))

class Low_Level_Weight_Space_Sampler(Gauss_Process_Sampler):
    def __init__(self, x, y, test_point_indices):
        super().__init__(x, y, test_point_indices)

    def fit(self):
        pass


