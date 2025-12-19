import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.cos(x)

plt.figure()
plt.plot(x, y)
plt.title("WSL sanity check plot")
plt.xlabel("x")
plt.ylabel("cos(x)")
plt.show()
