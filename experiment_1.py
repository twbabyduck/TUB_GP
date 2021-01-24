#plot infos
#left: function space <> exact
#middle function space view <> sparse
#right: weight space (basis functions)



#Figure 1 of Paper

import sampler
import numpy as np
import utils
import matplotlib.pyplot as plt


#x_test, y_test = utils.generate_data3(nr_train_data=15)
x_train, y_train = utils.generate_data4(nr_train_data = 150, shuffle=False)
x_test, y_test = utils.generate_data4(nr_train_data = 150, shuffle=True)

import pdb; pdb.set_trace()

s_left = sampler.Function_Space_Exact(x_train,y_train)
s_middle = sampler.Function_Space_Sparse_Sampler(x_train,y_train,8)
#s_right = sampler.Weight_Space_Sampler(x_train,y_train,l=2000)

samplers=[s_left,s_middle]


for sampler in samplers:
    print(sampler)
    sampler.sample_from_posterior(x_test,y_test)



