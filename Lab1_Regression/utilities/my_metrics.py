import numpy as np


def mse_test(pred,y_real):
    return np.sum((pred - y_real) ** 2) / len(y_real)


def rmse_test(pred,y_real):
    mse_test_res = mse_test(pred,y_real)
    return np.sqrt(mse_test_res)