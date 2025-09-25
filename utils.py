import os
import numpy as np
import pandas as pd

#matlab functions
def fullfile(*args):
    return os.path.join(*args)

def rot90(mat, k=1):
    return np.rot90(mat, k)

def writetable(df, file_path):
    df=pd.DataFrame(df)
    df.to_csv(file_path, index=False)

def intersect(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

## === 正規化（訓練統計のみでフィッティング→リーク防止）===
def normalize(Train_raw,Val_raw=None,dim=0):
#    print("Train_raw",Train_raw.shape)
    muX    = np.mean(Train_raw, dim)
    sigmaX = np.std(Train_raw, dim)
    sigmaX=sigmaX+1e-7

    Train = (Train_raw - muX) / sigmaX
    if(Val_raw is None):
        return Train,None,muX,sigmaX
    else:
        Val   = (Val_raw   - muX) / sigmaX
        return Train,Val,muX,sigmaX

def height(x):
    return x.shape[0]

def downsample(x, factor):
    return x[::factor]