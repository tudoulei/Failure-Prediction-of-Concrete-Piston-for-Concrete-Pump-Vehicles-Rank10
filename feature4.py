
# coding: utf-8

# In[9]:


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



dtypes_col=['活塞工作时长',
             '发动机转速',
             '油泵转速',
             '泵送压力',
             '液压油温',
             '流量档位',
             '分配压力',
             '排量电流',
             '低压开关',
             '高压开关',
             '搅拌超压信号'
             ,'正泵'
             ,'反泵'
           ]

dtypes_type = []

for i in range(len(dtypes_col)):
    dtypes_type.append('float16')
    
column_types = dict(zip(dtypes_col, dtypes_type))     
# column_types['设备类型']='int16'
# column_types['低压开关']='int8'
# column_types['高压开关']='int8'
# column_types['搅拌超压信号']='int8'
column_types['正泵']='int8'
column_types['反泵']='int8'

train = pd.read_csv('new_data/data_train.csv',dtype=column_types)
test = pd.read_csv('new_data/data_test.csv',dtype=column_types)

sample = pd.read_csv('new_data/submit_example.csv')


# In[ ]:


from sklearn import preprocessing
enc = preprocessing.LabelEncoder()
categorical_columns = ['设备类型']
for f in categorical_columns:
#     try:
    data =pd.concat([train,test],sort=False)
    enc.fit(data[f].values.reshape(-1, 1))
    train[f] = enc.fit_transform(train[f])
    test[f] = enc.fit_transform(test[f])

"save device "
datashebei_train  = train[f]
datashebei_test   = test[f]


# In[ ]:


def signal_is_1(x):
    return np.sum(x == 1)
def signal_is_0(x):
    return np.sum(x == 0)

from scipy import signal

def peak_num_1(x):
    return len(signal.find_peaks_cwt(x, np.arange(1,50)))

def peak_num_2(x):
    return len(signal.find_peaks_cwt(x, np.arange(1,2)))


# In[ ]:


def process(data):
    data_all  = data['sample_file_name'].groupby(data['sample_file_name']).count().rename('长度')
    data_all = pd.DataFrame(data_all)

    data_shebeileixing = data.groupby(data['sample_file_name']).mean()['设备类型']
    data.drop('设备类型',axis=1,inplace=True)
    
    is_01_feature =['低压开关', '高压开关', '搅拌超压信号', '正泵', '反泵','sample_file_name']
    need_groupby_fea = [f for f in data.columns if f not in is_01_feature]
    

    print('mean')
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name']).mean().rename(columns=lambda x:x+'_mean')],axis=1)
    print('max')
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])[need_groupby_fea].max().rename(columns=lambda x:x+'_max')],axis=1)
    print('min')
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])[need_groupby_fea].min().rename(columns=lambda x:x+'_min')],axis=1)
    print('sum')
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])[need_groupby_fea].sum().rename(columns=lambda x:x+'_sum')],axis=1)
    print('std')
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name']).std().rename(columns=lambda x:x+'_std')],axis=1)
    print('median')
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])[need_groupby_fea].median().rename(columns=lambda x:x+'_median')],axis=1)
    
    "统计信号"
    print("统计信号")
    print("计算信号==1")
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])['低压开关', '高压开关', '搅拌超压信号', '正泵', '反泵'].agg(signal_is_1).rename(columns=lambda x:x+'_signal_is_1')],axis=1)
    print("计算信号==0")
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])['低压开关', '高压开关', '搅拌超压信号', '正泵', '反泵'].agg(signal_is_0).rename(columns=lambda x:x+'_signal_is_0')],axis=1)
    
    print("peak_num_1")
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])['搅拌超压信号'].agg(peak_num_1).rename(columns=lambda x:x+'_peak_num_1')],axis=1)
    print("peak_num_2")
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])['正泵','反泵'].agg(lambda x: len(signal.find_peaks_cwt(x, [1,2]))).rename(columns=lambda x:x+'_peak_num_2')],axis=1)
    print("peak_num_3")
    data_all = pd.concat([data_all,data.groupby(data['sample_file_name'])['发动机转速',  '泵送压力', '液压油温', '流量档位', '分配压力', '排量电流',].agg(lambda x: len(signal.find_peaks_cwt(x, [1,2,3,4,5,6,7,8,9,10]))).rename(columns=lambda x:x+'_peak_num_2')],axis=1)


    print('设备类型')
    data_all = pd.concat([data_all,data_shebeileixing],axis=1)   
    
    print('feature over.................')
    
    print("加减乘除特征")
    data_all['活塞工作时长/长度'] = data_all['活塞工作时长_mean'] / data_all['长度']
    data_all['油泵转速/长度'] = data_all['油泵转速_mean'] / data_all['长度'] 
    data_all["f0"]=  data_all["发动机转速_mean"]-data_all["油泵转速_mean"]
    data_all["f2"] = data_all['正泵_signal_is_1'] * data_all['长度']
    data_all["f5"] = data_all['活塞工作时长_mean'] * data_all['长度']
    data_all["f6"] = (data_all['正泵_signal_is_1']+data_all['反泵_signal_is_1'])/data_all['长度']
    data_all["f7"]=  data_all["f6"]*data_all['活塞工作时长_mean']
    
    return data_all


data_train = process(train)
data_test = process(test)


# In[ ]:
data_train = pd.concat([data_train,train.groupby(train['sample_file_name'])['发动机转速','泵送压力', '液压油温', '流量档位', '分配压力', '排量电流'].agg(np.ptp).rename(columns=lambda x:x+'_ptp').reset_index(drop=True)],axis=1)
data_test = pd.concat([data_test,test.groupby(test['sample_file_name'])['发动机转速','泵送压力', '液压油温', '流量档位', '分配压力', '排量电流'].agg(np.ptp).rename(columns=lambda x:x+'_ptp').reset_index(drop=True)],axis=1)


data_train.to_csv('mydata/data_train.csv')
data_test.to_csv('mydata/data_test.csv',index=None)
