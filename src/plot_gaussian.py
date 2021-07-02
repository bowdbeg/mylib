import numpy as np
import matplotlib.pyplot as plt

mu = 0.0
sigma = 1.

x = np.linspace(-3.0, 3.0, 100)
y = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)

plt.plot(x,y,color='black')
plt.fill_between(x, y, np.zeros_like(y),color='#add8e6')
plt.tick_params(bottom=False, left=False, right=False, top=False)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.ylim(0.,0.5)
# plt.show()
# plt.savefig('gaussian.pdf')
plt.savefig('gaussian.svg')