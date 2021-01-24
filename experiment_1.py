#plot infos
#left: function space <> exact
#middle function space view <> sparse
#right: weight space (basis functions)



#Figure 1 of Paper

import sampler
import numpy as np
import utils

x_test, y_test = utils.generate_data(nr_train_data=1024)
x_train, y_train = utils.generate_data(nr_train_data = 4)

s_left = sampler.Function_Space_Exact(x_train,y_train)
s_middle = sampler.Function_Space_Sparse_Sampler(x_train,y_train,8)
s_right = sampler.Weight_Space_Sampler(x_train,y_train,l=2000)

samplers=[s_right,s_left,s_middle]
samplers=[s_left,s_middle]

for sampler in samplers:
    print(sampler)
    sampler.sample_from_posterior(x_test,y_test)


