import matplotlib.pyplot as plt
import numpy as np

t1 = np.arange(-30.0, 30.0, 0.01)
s1 = np.exp(-1*(t1)**2 / 2) / np.sqrt(2*np.pi)
plt.plot(t1, s1, label = 'Mean = 0, S.D. = 1')

t2 = np.arange(-30.0, 30.0, 0.01)
s2 = np.exp(-1*(t2 - 5)**2 / 50) / np.sqrt(2*np.pi*25)
plt.plot(t2, s2, label = 'Mean = 5, S.D. = 5')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Distribution')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.savefig("Gaussian.png")
plt.show()
