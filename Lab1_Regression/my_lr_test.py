import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,LabelEncoder




from utilities.my_metrics import mse_test,rmse_test



data_root="D:\TJCS\ML&DL\datasets\\boston"
# house-prices-advanced-regression-techniques"
train_file="boston.csv"#"train.csv"
val_file="val.csv"
test_file=None#"test.csv"


df_train = pd.read_csv(data_root+"\\"+train_file) if train_file else None
df_test =  pd.read_csv(data_root+"\\"+test_file) if test_file else None
# print(df_train)
# print(df_test)

df_all = pd.concat([df_train, df_test])  # 给所有数据编码

PRD_LABEL="MEDV" # 预测标签
# PRD_LABEL="SalePrice" # 预测标签

feats=list(df_all.columns)
feats.remove(PRD_LABEL)


df_train_enc =df_all[df_all[PRD_LABEL].notna()].reset_index(drop=True)



# df_train_X= df_train.drop([PRD_LABEL], axis=1)
df_test_enc = df_all[df_all[PRD_LABEL].isna()].reset_index(drop=True)

# print(df_train_enc)
X=df_train_enc[feats]
y=df_train_enc[PRD_LABEL]
# y=np.log1p(df_train_enc[PRD_LABEL])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state= 0)

scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from classifiers.my_regressionModels import MyLinearRegression

my_lr=MyLinearRegression()
# my_lr.fit(X_train,y_train)
my_lr.train(X_train,y_train
            ,learning_rate=0.01,iterations=5000,batch_size=X_train.shape[0],verbose=True)
my_lr_pre = my_lr.predict(X_test)

print(my_lr_pre)

y_ori=y_test
print("mylr R2=",r2_score(y_ori,my_lr_pre ))#模型评价, 决定系数
print(rmse_test(my_lr_pre,y_ori))