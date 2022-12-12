import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def normalize_data(x):
   x = np.nan_to_num(x)
   x_normed = (x - np.mean(x, axis=(0,1), keepdims=True))/(np.std(x, axis=(0,1), keepdims=True)  + 0.0000001)
   return x_normed

def Catergorical2OneHotCoding(a):
    b = np.zeros((a.size, np.max(a) + 1))
    b[np.arange(a.size), a] = 1
    return b

if __name__ == "__main__":
    
    print("Fine")
