import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
import numpy as np


data = pd.read_csv("sensor_data_700.txt", delimiter = " ",
	header=None, names=("date", "time", "ir", "lidar"))


d = data[ (data["time"]<160000) & (data["time"]>=120000) ]

d = d.loc[:, ["ir", "lidar"]]	# extract only ir and lidar data

#sns.jointplot(d["ir"], d["lidar"], d, kind="kde")
#plt.show()

print("光センサの計測値の分散:", d.ir.var())
print("LiDARの計測値の分散:", d.lidar.var())

diff_ir = d.ir - d.ir.mean()
diff_lidar = d.lidar - d.lidar.mean()

a = diff_ir * diff_lidar

print("共分散:", sum(a)/(len(d)-1))

d.mean()
d.cov()
print(d.cov())

irlidar = multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)

x, y = np.mgrid[0:40, 710:750]
pos = np.empty(x.shape + (2,))
pos[:,:,0] = x
pos[:,:,1] = y
#cont=plt.contour(x,y,irlidar.pdf(pos))
#cont.clabel(fmt='%1.1e')



print("X座標：",x)
print("Y座標：",y)

c = d.cov().values + np.array([[0, 20], [20, 0]])
tmp = multivariate_normal(mean = d.mean().values.T, cov = c)
cont = plt.contour(x,y, tmp.pdf(pos))
plt.show()