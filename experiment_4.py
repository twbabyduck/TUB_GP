from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sampler
import utils
import warnings
warnings.filterwarnings("ignore")


def ackley_1d(x, y=0):
    ackley = (-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2)))
              - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
              + np.e + 20)
    return ackley


# clearing past figures
plt.close('all')
plt.figure(figsize=[10, 4], dpi=150)

# data
X = np.linspace(-4, 4, 500)
Y = ackley_1d(X)

# plotting
plt.plot(X, Y, 'k--', linewidth=2)
plt.title("""Ackley's function at $y=0$""", fontsize=14)
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.show()


# let us draw 20 random samples of the Ackley's function
x_observed = np.random.uniform(-4, 4, 20)
y_observed = ackley_1d(x_observed)

# let us use the Matern kernel
K = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

# instance of GP
gp = GaussianProcessRegressor(kernel=K)

# fitting the GP
gp.fit(x_observed.reshape(-1, 1), y_observed)


# let us check the learned model over all of the input space
X_ = np.linspace(-4, 4, 500)
y_mean, y_std = gp.predict(X_.reshape(-1, 1), return_std=True)

# clearing past figures
plt.close('all')
plt.figure(figsize=[10, 4], dpi=150)

# data
Y = ackley_1d(X_)

# plotting
plt.plot(X_, Y, 'k--', linewidth=2, label='Actual function')
plt.plot(x_observed, y_observed, 'bo',
         label="""Random Samples of Ackley's function""", alpha=0.7)
plt.plot(X_, y_mean, 'r', linewidth=2,
         label='Gaussian Process mean', alpha=0.7)
plt.fill_between(X_, y_mean - y_std, y_mean + y_std, alpha=0.2, color='r')
plt.title("""Ackley's function at $y=0$, GP fit with random samples""", fontsize=14)
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()

# function samples
y_samples = gp.sample_y(X_.reshape(-1, 1), 10)

# clearing past figures
plt.close('all')
plt.figure(figsize=[10, 4], dpi=150)

# plotting random observations of function
plt.plot(x_observed, y_observed, 'bo',
         label="""Random Samples of Ackley's function""", alpha=1.0)
plt.plot(X_, Y, 'k--', linewidth=2, label='Actual function')


# plotting all the posteriors
for posterior_sample in y_samples.T:
    plt.plot(X_, posterior_sample,
             label="""Random Samples of Ackley's function""", alpha=0.7, linewidth=1)

# title and labels
plt.title("""Ackley's function at $y=0$, GP posterior samples""", fontsize=14)
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.show()


# function to create an animation with the visualization
def ts_gp_animation_ackley(ts_gp, max_rounds):

    round_dict = {}

    for round_id in range(max_rounds):

        # recording all the info
        X_observed, y_observed, X_grid, posterior_sample, posterior_mean, posterior_std = ts_gp.choose_next_sample()

        # adding to dict
        round_dict[round_id] = {'X_observed': X_observed,
                                'y_observed': y_observed,
                                'X_grid': X_grid,
                                'posterior_sample': posterior_sample,
                                'posterior_mean': posterior_mean,
                                'posterior_std': posterior_std}

    fig, ax = plt.subplots(figsize=[10, 4], dpi=150)

    # plotting first iteration
    ax.plot(X, Y, 'k--', linewidth=2, label='Actual function')
    ax.plot(round_dict[0]['X_observed'], round_dict[0]['y_observed'],
            'bo', label="""GP-Chosen Samples of Ackley's function""", alpha=0.7)
    ax.plot(round_dict[0]['X_grid'], round_dict[0]['posterior_sample'],
            'r', linewidth=2, label='Sample from the posterior', alpha=0.7)
    ax.fill_between(round_dict[0]['X_grid'], round_dict[0]['posterior_mean'] - round_dict[0]['posterior_std'],
                    round_dict[0]['posterior_mean'] + round_dict[0]['posterior_std'], alpha=0.2, color='r')
    plt.title("""Thompson Sampling Gaussian Process""", fontsize=14)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    ax.set_ylim(-0.5, 10)

    # function for updating
    def animate(i):
        ax.clear()
        ax.plot(X, Y, 'k--', linewidth=2, label='Actual function')
        ax.plot(round_dict[i]['X_observed'], round_dict[i]['y_observed'],
                'bo', label="""GP-Chosen Samples of Ackley's function""", alpha=0.7)
        ax.plot(round_dict[i]['X_grid'], round_dict[i]['posterior_sample'],
                'r', linewidth=2, label='Sample from the posterior', alpha=0.7)
        ax.fill_between(round_dict[i]['X_grid'], round_dict[i]['posterior_mean'] - round_dict[i]['posterior_std'],
                        round_dict[i]['posterior_mean'] + round_dict[i]['posterior_std'], alpha=0.2, color='r')
        plt.title("""Thompson Sampling Gaussian Process""", fontsize=14)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        ax.set_ylim(-0.5, 10)
        return ()

    # function for creating animation
    anim = FuncAnimation(fig, animate, frames=max_rounds,
                         interval=500, blit=True, repeat=True)
    plt.show()


ts_gp = sampler.Thompson_Sampler(
    n_random_samples=2, objective=ackley_1d, x_bounds=(-4, 4))

# showing animnation
ts_gp_animation_ackley(ts_gp, 20)
