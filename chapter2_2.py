import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sensor_data_200.txt", delimiter=" ", header=None, names=("data", "time", "ir", "lidar"))
mean1 = sum(data["lidar"].values)/len(data["lidar"].values)

mean2 = data["lidar"].mean()

print(mean1, mean2)

zs = data["lidar"].values

mean = sum(zs)/len(zs)
diff_square = [ (z - mean)**2 for z in zs ]


sampling_var = sum(diff_square)/(len(zs))
unbiased_var = sum(diff_square)/(len(zs) - 1)

import math

stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)

from scipy.stats import norm

zs = range(190, 230)
ys = [norm.pdf(z, mean1, stddev1) for z in zs]

ys = [norm.cdf(z+0.5, mean1, stddev1) - norm.cdf(z-0.5, mean1, stddev1) for z in zs]

#plt.bar(zs, ys)
#plt.show()


import random

samples=[ random.choice([1, 2, 3, 4, 5, 6]) for i in range(10000)]
print(sum(samples)/len(samples))
