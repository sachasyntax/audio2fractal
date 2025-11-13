import numpy as np
import matplotlib.pyplot as plt

N = 20
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
z = np.linspace(-1.5, 1.5, N)
X, Y, Z = np.meshgrid(x, y, z)

values = np.sin(X**2 + Y**2 + Z**2 * 3) * 50 + 50  

x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()
c_flat = values.flatten()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x_flat, y_flat, z_flat, c=c_flat, cmap='plasma', s=20)
plt.colorbar(sc, shrink=0.5)
plt.show()
