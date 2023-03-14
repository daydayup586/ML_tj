import numpy as np

from utilities.my_Func import *

class MyLinearRegression(object):
    def __init__(self):
        self.W = None
        self.b = None
        self.Theta = None # W与b拼接


    def fit(self,X,y):
        """
        正规方程直接解得所有theta

        Inputs:
        - X: (N, D) 
        - y: (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        

        Returns:
        """
        bX = np.hstack([np.ones((X.shape[0],1)), X]) #  [b(theta0),X]
        self.Theta = np.linalg.inv(bX.T.dot(bX)).dot(bX.T).dot(y) #(𝑋^𝑇*𝑋)^(−1)*𝑋^𝑇*𝑦
        self.b = self.Theta[0]
        self.W = self.Theta[1:]

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        iterations=100,
        batch_size=200,
        verbose=False,
    ):
        """
        训练的方式更新参数

        Inputs:
        - X: (N, D) 
        - y: (N,) 
        

        Returns:
        """
        data_num, feature_dim = X.shape


        if self.W is None:
            self.W = 100*np.random.randn(feature_dim) 
        if self.b is None:
            self.b = 100*np.random.randn(1) 
        
        X=np.array(X)
        y=np.array(y)

        # sgd 优化
        # loss_history = []
        for it in range(iterations):

            train_indices=np.random.choice(range(data_num),batch_size,replace=True)
            X_batch= X[train_indices]
            # print(y.shape)
            y_batch=y [train_indices]
            # print('X_batch.shape: ',X_batch.shape,' ','y_batch.shape: ',y_batch.shape)



     
            cost,grad=lr_cost_grad_ori(X_batch,y_batch,self.W,self.b)


            self.W-=learning_rate*grad['dW']
            self.b-=learning_rate*grad['db']
            


            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, iterations, cost))

        # return loss_history

    def predict(self, X):
        """
        根据训练好或直接计算出的最优W,b预测结果

        Inputs:
        - X: (N, D) 

        Returns:
        - y_pred
        """
      
        y_pred=X.dot(self.W)+self.b
        
        return y_pred

    def get_cost_grad(self, X_batch, y_batch, reg):
        """
        TODO: 可以用子类覆写

        
        """
        cost,grad=lr_cost_grad_ori(X_batch,y_batch,self.W,self.b)
        return cost,grad

        pass


'''岭回归'''
class MyRidgeRegression(object):
    
    pass


'''lasso回归'''
class MyLassoRegression(object):

    pass