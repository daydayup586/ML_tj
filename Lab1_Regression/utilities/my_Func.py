import numpy as np

def lr_cost_grad_ori(X, y, W, b): 
    """
    多元线性回归的代价和梯度
    
    Inputs:
    - X (N,D)
    - y (N,)
    - W (D,)
    - b (1,)
    
    Returns:
    - cost
    - grad: dict
      - dW
      - db

    """

    grad={}


    N = X.shape[0]
    residual=X.dot(W)+b-y
    cost=np.mean((residual)**2)/2
    dW= np.mean(residual.reshape(N,1)*X,axis=0) # 纵轴平均
    db= np.mean(residual)

    grad['dW'],grad['db']=dW,db

    return cost,grad



def lr_grad_ori(X, y, W, b): 
    """
    多元线性回归的梯度
    
    Inputs:
    - X (N,D)
    - y (N,)
    - W (D,)
    - b (1,)
    
    Returns:
    - grad

    """
    N,D=X.shape


    dW/=N
    db/=N
    return dW,db
# def :
