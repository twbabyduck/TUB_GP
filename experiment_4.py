samplers = [
    sampler.Thompson_Sampler(x, y, [1], {"paralell": true, "type": "random_search"}),
    sampler.Thompson_Sampler(x, y, [1], {"paralell": true, "type": "dividing_rectangles"}),
    sampler.Thompson_Sampler(x, y, [1], {"paralell": true, "type": "function_space"}),
    sampler.Thompson_Sampler(x, y, [1], {"paralell": true, "type": "weight_space"}),
    sampler.Thompson_Sampler(x, y, [1], {"paralell": true, "type": "decoupled_sampling"}),
]

results=[]
configs = [[2, 1024], [8, 1024], [8, 4096]]
n_evaluations = [1, 16, 32, 48, 64, 256, 512, 768, 1024]
for config in configs:
    d = config[0]
    l = config[1]

    for n in n_evaluations:
        for sampler in samplers:
            sapmpler.fit()
            regret=sampler.regret()
            #smart dict push to results


plot_all()