from itertools import combinations
import numpy as np

def getPolyFeatures(feats_in,mode ='selfPow',max_pow=3):
    """
        构造多项式特征

        Inputs:
        - feats_in: (N, D) 
        - mode: 默认是"selfPow"即只有自己的平方、立方等 | "interOnly"只有【无重复组合】交叉项，即ab,ac,bc | "both"前两种都有
        - max_pow: 最大的次数，默认是2次

        Returns:
        - feats_out:(N, ___) 
    """
    
    feats_out=np.array(feats_in)
    # print(feats_out.shape)
    N, D = feats_out.shape

    if mode =='selfPow':
        for i in range(2,max_pow+1):
            feats_out=np.hstack((feats_out,np.power(feats_in,i)))
    elif mode=='interOnly':   
        for i in range(2,max_pow+1):
            c = list(combinations(range(D),i))
            for t in c:
                tmp=np.prod(feats_out[:,np.array(t)],axis=1).reshape(-1,1)
                # print(tmp.shape)
                feats_out=np.hstack((feats_out,tmp))
    elif mode=='both':
        for i in range(2,max_pow+1):
            feats_out=np.hstack((feats_out,np.power(feats_in,i)))
        for i in range(2,max_pow+1):
            c = list(combinations(range(D),i))
            for t in c:
                tmp=np.prod(feats_out[:,np.array(t)],axis=1).reshape(-1,1)
                # print(tmp.shape)
                feats_out=np.hstack((feats_out,tmp))



    return feats_out
