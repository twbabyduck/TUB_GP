import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial

# Define RBF (radial basis function kernel) 
def radial_basis_function_kernel(x_1, x_2, sigma = 1, length = 1):
    """
    radial_basis_function_kernel with σ = 1
    You can change different value of (1)variance and (2)length to see the difference
    """
    variance = sigma ** 2
    # squared Euclidean distance (i.e. L2 distance)
    sqeuclidean = -(0.5/length**2) * scipy.spatial.distance.cdist(x_1, x_2, 'sqeuclidean')
    return variance * np.exp(sqeuclidean)

from matplotlib import cm

# Fake Data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=150)
xlim = (-5, 5)
X = np.expand_dims(np.linspace(*xlim, 25), 1)
Σ = radial_basis_function_kernel(X, X, length=0.8)

# Plot Covariance Matrix
cov = ax1.imshow(Σ, cmap=cm.YlGnBu)
cbar = plt.colorbar(
    cov, ax=ax1, fraction=0.045, pad=0.05)
cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
ax1.set_title((
    'RBF Kernel \n'
    'covariance matrix figure'))
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('x', fontsize=13)
ticks = list(range(xlim[0], xlim[1]+1))
ax1.set_xticks(np.linspace(0, len(X)-1, len(ticks)))
ax1.set_yticks(np.linspace(0, len(X)-1, len(ticks)))
ax1.set_xticklabels(ticks)
ax1.set_yticklabels(ticks)
ax1.grid(False)

# Fake Data
xlim = (-5, 5)
X = np.expand_dims(np.linspace(*xlim, num=100), 1)
zero = np.array([[0]])
Σ0 = radial_basis_function_kernel(X, zero, length=0.8)

# Plots
ax2.plot(X[:,0], Σ0[:,0], label='$k(x,0)$')
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel('covariance', fontsize=13)
ax2.set_title((
    'RBF Kernel covariance\n'
    'between $x$ and $0$'))
ax2.set_xlim(*xlim)
ax2.legend(loc=1)

fig.tight_layout()
plt.show()