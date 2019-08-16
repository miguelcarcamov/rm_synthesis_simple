import numpy as np

def MSE(x, y):
    dif = x-y
    mse = np.sum(dif**2)/len(x)
    return mse

def RMSE(x, y):
    mse = MSE(x, y)
    rmse = np.sqrt(mse)
    return rmse

def SNR(x,y):
    dif = x-y
    num = np.sum(x**2)
    den = np.sum(dif**2)
    snr = 10.0*np.log10(num/den)
    return snr

def CompressibilityRatio(x):
    nzero = np.count_nonzero(x)
    return nzero/len(x)

def coherence(A):
    rows = A.shape[0]
    columns = A.shape[1]
    max = 0

    for i in range(columns):
        for j in range(i + 1, columns):
            col_a = A[:,i]
            col_b = A[:,j]
            num = np.abs(np.dot(col_a, col_b))
            den = np.linalg.norm(col_a,2)*np.linalg.norm(col_b,2)
            new_max = num/den
            if new_max > max:
                max = new_max
    return new_max
