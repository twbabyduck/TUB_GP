# Figure 3 of Paper
x, y = utils.generate_data()
x, y = utils.shuffle_data(x, y)

s1 = sampler.Weight_Space_Sampler(x, y, [1]),
s2 = sampler.Decoupled_Sampler(x, y, [1])


total_results=[]
outer_loop = [1]  # What is the x axis description? Don't get this in Figure 3
for nr_training_locations in outer_loop:
    for inner_train_location in [2 ** 2, 2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10]:
        for l in [1024,4096,16384]:
            x = x[0:inner_train_location]
            y = y[0:inner_train_location]
            s1.fit()
            s1.wasser_distance
            #save to total_results


plot_everything(total_results)


Question: Weight Space Sampling is part of decoupled or not

