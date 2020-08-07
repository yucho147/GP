import numpy as np
np.random.seed(14)

num = 200
sample_num = 7
idx = np.random.randint(0, num, sample_num)

x = np.linspace(0, 1, num)
y = np.random.normal(np.sin(x * 20.) * 10., 0.1) + 10.

data = np.array([x[idx], y[idx]]).T
np.savetxt('./sample_data_1.csv', data, delimiter=',', header='x,y', comments='')
