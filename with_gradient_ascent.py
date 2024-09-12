import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define constants
ITERS = 100000
LEARNING_RATE = .001

v = np.array([100, 50, 60, 30, 30, 70, 40, 60, 50, 30]).astype(np.float32)
t = np.array([51, 22, 17, 19, 13, 30, 24, 19, 24, 12]).astype(np.float32)
assert len(v) == len(t)
n = len(v)
s = 2
c = 50

x = np.ones(10).astype(np.float32) * 5

# Unless otherwise stated, all the following take vector arguments (except c & s)
def prob(x, t):
    prob_instance = lambda x, t: x / (x + t)
    return np.array(list(iter.starmap(prob_instance, zip(x, t))))

def pay(x, v, s):
    pay_instance = lambda x, v, s: v - s * x
    return np.array(list(iter.starmap(pay_instance, zip(x, v, [s] * n))))

def f(x, t, v, s):
    return np.dot(prob(x, t), pay(x, v, s))

def g(x, c):
    return sum(x) - c

def r(x, t, v, s, c):
    return f(x, t, v, s) - g(x, c) ** 2

def gradient_f(x, t, v, s):
    gradient_instance = lambda x, t, v, s: (v * t - 2 * s * x * t - s * x ** 2) / (t + x) ** 2
    return np.array(list(iter.starmap(gradient_instance, zip(x, t, v, [s] * n))))

def gradient_r(x, t, v, s, c):
    return gradient_f(x, t, v, s) - 2 * g(x, c)

def optimal(t, v, s):
    return (2 * s * t - (4 * s ** 2 * t ** 2 + 4 * v * t * s) ** 0.5) / -2 * s

unconstrained = np.array(list(iter.starmap(optimal, zip(t, v, [s] * n))))
unconstrained /= sum(unconstrained) / 50
print(f"The new sum = {sum(unconstrained)}")
x = unconstrained.copy()

# Ascend!
learning_curve = np.zeros(ITERS)
for i in tqdm(range(ITERS)):
    learning_curve[i] = r(x, t, v, s, c)
    x += gradient_r(x, t, v, s, c) * LEARNING_RATE
    # x[x < 0] = 0

print("Final answer:")
print(x)
print(f"This is a total of {sum(x)} tickets!")

plt.plot(x)
plt.plot(unconstrained)
# plt.plot(learning_curve)
plt.show()