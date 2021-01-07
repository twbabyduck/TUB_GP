# Figure 2 of Paper

import sampler
import numpy as np
import utils

x, y = utils.generate_data()

decoupled_sampler = sampler.Decoupled_Sampler(x, y, [1]),


#Left Plot:
some_plots=decoupled_sampler.fit_prior()


#Middel plot
some_other_plots=decoupled_sampler.fit_update()
