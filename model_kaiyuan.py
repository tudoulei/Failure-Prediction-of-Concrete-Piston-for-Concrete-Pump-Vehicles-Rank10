
# coding: utf-8

# In[1]:


import time
import os.path
import os
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from pyecharts import Line, Grid
import numpy as np
import pandas as pd
 # 内部测试
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import xgboost as xgb
from lightgbm import LGBMRegressor
import math
import warnings
import numpy as np 
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签\n"
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号\n"


# In[2]:


data_train=pd.read_csv('mydata/data_train_kaiyaun.csv')

data_test=pd.read_csv('mydata/data_test_kaiyaun.csv')


# In[3]:


# 删除某一类别占比超过90%的列
good_cols = list(data_train.columns)
bad_cols = []
for col in data_train.columns:
    rate = data_train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate >0.9:
        bad_cols.append(col)
        good_cols.remove(col)
        print(col,rate)




print("删除 明显无用特征")
for i in bad_cols:
    try:
        data_train.drop(i,inplace=True,axis=1)
        data_test.drop(i,inplace=True,axis=1)
    except:
        print(i)


# In[5]:



data_train.drop('sample_file_name',inplace=True,axis=1)


# In[6]:


del_feature = []
for i in data_train.columns:
    if data_train[i][data_train[i]==np.inf].shape[0] >0:
        del_feature.append(i)
        print(i)

print("删除 明显无用特征")
for i in del_feature:
    try:
        data_train.drop(i,inplace=True,axis=1)
        data_test.drop(i,inplace=True,axis=1)
    except:
        print(i)


# In[7]:


print("删除 明显无用特征")
for i in ['活塞工作时长/长度','活塞工作时长_sum']:
    try:
        data_train.drop(i,inplace=True,axis=1)
        data_test.drop(i,inplace=True,axis=1)
    except:
        print(i)


# In[8]:


do_test=1

if do_test==1:
    X_test = data_test.values
X_train = data_train.values

from sklearn.model_selection import StratifiedKFold
nfd=0.455
def eval_score(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.where(preds>nfd,1,0)
    score = f1_score(preds,labels,average='macro')
    return 'score', score, True
if do_test==1:
    print(X_train.shape,X_test.shape)

dele_label=1
if dele_label:
    label = pd.read_csv('new_data/train_labels.csv')
    print(label.groupby(label['label']).count())
    y_train = label['label'].values

param = {
    'num_leaves': 30,
         'min_data_in_leaf': 10, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,  # 没影响
         "boosting": "gbdt",
    
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 917,
#          "metric": ['auc','mse'],
         "lambda_l1": 0.1,# 调成0.3试试
#         "lambda_l2":0.1,
         'nthread': 2,
#          'device': 'gpu',
         "verbosity": -1}
# StratifiedKFold
folds = KFold(n_splits=5, shuffle=True, random_state=250)
oof_lgb = np.zeros(X_train.shape[0])
if do_test==1:
    predictions_lgb = np.zeros(X_test.shape[0])
#     predictions_lgb=[]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, 
                    trn_data,
                    num_round, 
                    valid_sets = [trn_data, val_data], 
                    feval=eval_score,
                    verbose_eval=200, 
                    early_stopping_rounds = 200)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    

    if do_test==1:
        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: ",f1_score(np.where(oof_lgb>nfd,1,0),y_train,average='macro'))

predictions_lgb=predictions_lgb[1:]
# In[11]:


upup =pd.DataFrame()
# pre_label = label = list(map(lambda x:1 if x>ll[np.argmax(auc)] else 0,predictions_lgb))
pre_label = label = list(map(lambda x:1 if x>0.4525 else 0,predictions_lgb))
upup['sample_file_name'] = pd.read_csv('new_data/submit_example.csv')['sample_file_name'] 
up_per = pd.DataFrame()
up_per['sample_file_name'] = upup['sample_file_name'] 
file_name ='复现数据'
try:
    os.mkdir(file_name)
except:
    print('已存在')
upup['label'] =pre_label
up_per['label'] =predictions_lgb

name = 'baseline-lamda=0.1-num=10-leaf10-kaiyuandetezheng'

up_per.to_csv(file_name+'/概率'+name+'.csv',index=None)