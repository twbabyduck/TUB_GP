#Figure 1 of Paper

import sampler
import numpy as np
import utils

x,y = utils.generate_data()
samplers=utils.generate_samplers(x,y)

samplers = samplers[0:1]

def exp1(nr_observations, inducing_locations, sampler):
    sampler.fit()
    sampler.plot()
    #sampler.reset()
    return sampler.plot(), sampler.wasserstein_distance()


for sampler in  samplers:
    
    plot,distance = exp1(4, None, sampler)
    print(distance)
    plot,distance = exp1(100, None, sampler)
    print(distance)

    plot,distance = exp1(4, 8, sampler)
    print(distance)
    plot, distance = exp1(100, 8, sampler)
    print(distance)

