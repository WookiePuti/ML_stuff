import math
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def create_dataset(n, varaince, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(n):
        y = val+random.randrange(-varaince, varaince)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]


    return np.array(xs, dtype = np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope(xs, ys):
    m = (mean(xs) * mean(ys) - mean(xs*ys)) / ((mean(xs))**2 - mean(xs**2))
    b = mean(ys)- m*mean(xs)
    return m, b


def squared_error(ys_orig, ys_reg):
    return sum((ys_reg-ys_orig)**2)


def coeff_of_determination(ys_orig, ys_reg):
    y_mean_reg = [mean(ys_orig) for y in ys_orig]
    squarred_error_reg = squared_error(ys_orig, ys_reg)
    squarred_error_y_mean = squared_error(ys_orig, y_mean_reg)
    return 1-(squarred_error_reg/squarred_error_y_mean)


xs, ys = create_dataset(40, 80, 2, correlation='pos')


m, b = best_fit_slope(xs, ys)

regression_y = m*xs + b

r_squared = coeff_of_determination(ys, regression_y)
print(r_squared)

plt.figure()
plt.plot(xs, regression_y)
plt.scatter(xs, ys)
plt.show()