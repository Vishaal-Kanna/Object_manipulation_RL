import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

success_grid = np.load('./success_grid.npy')

x = success_grid[:,0]
y = success_grid[:,1]
z = success_grid[:,2]
ax.scatter3D(x[success_grid[:,3]==1], y[success_grid[:,3]==1], z[success_grid[:,3]==1], s = 1, color = "green")
ax.scatter3D(x[success_grid[:,3]==0], y[success_grid[:,3]==0], z[success_grid[:,3]==0], s = 1, color = "red")

ax.scatter3D(-0.45, 0.0, 0.0, s = 500, color = "blue")
plt.show()

# print(success_grid)