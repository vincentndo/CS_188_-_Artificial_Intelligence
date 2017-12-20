import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
	x1, x2 = 1, 1
	w11, w12, w21, w22 = -4, -4, -2, -2
	wy1, wy2 = -5, -3

	h1 = sigmoid(w11 * x1 + w12 * x2 + 0.5)
	h2 = sigmoid(w21 * x1 + w22 * x2 + 0.5)
	y = round( sigmoid(wy1 * h1 + wy2 * h2 + 0.5) )
	print(y)
