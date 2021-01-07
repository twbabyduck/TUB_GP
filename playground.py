import tensorflow as tf
from gpflow.config import default_float as floatx
from gpflow.kernels import Matern52
from gpflow_sampling.models import PathwiseGPR
from gpflow_sampling.sampling.updates import cg as cg_update

import sampler
import numpy as np
import utils

x,y = utils.generate_data()
print(x)
print(y)

dummy_sampler = sampler.Dummy_Gauss_Sampler([1,100,2,2],[1,100,2,2],[1,2,3])

dummy_sampler.fit()