import matplotlib.pyplot as plt
import numpy as np

t = np.arange(-10.0, 10.0, 0.01)
s = np.exp(-1*(t - 5)**2 / 50) / np.sqrt(2*np.pi)
plt.plot(t, s)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Distribution (Mean = 0, S.D. = 1)')
plt.grid(True)
plt.savefig("test.png")
plt.show()
