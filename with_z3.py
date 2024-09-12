from z3 import *

# Define constants
v = [100, 50, 60, 30, 30, 70, 40, 60, 50, 30]
t = [51, 22, 17, 19, 13, 30, 24, 19, 24, 12]
assert len(v) == len(t)
n = len(v)
s = 2
c = 50

# Initialize vector x
x = [Real(f"x{i}") for i in range(n)]
gradient_x = [(v[i] * t[i] - 2 * s * x[i] * t[i] - s * x[i] ** 2) / (t[i] + x[i]) ** 2 for i in range(n)]

# l for lambda
l = Real("l")

solve(*[gradient_x[i] == l for i in range(n)], *[x[i] > 0 for i in range(n)], sum(x) == c)
