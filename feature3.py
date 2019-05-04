
# coding: utf-8

# 制作关于异常点的特征集


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
# pd.set_option('display.max_colwidth',1000)
# pd.set_option('display.height',1000)
# pd.set_option('display.max_rows',500)
# pd.set_option('display.max_columns',500)
# pd.set_option('display.width',1000)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签\n"
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号\n"


# In[ ]:


import pandas as pd
import os
from tqdm import *
# -*- coding: utf-8 -*-
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal
data_list = os.listdir('new_data/test/')
df = pd.read_csv('new_data/test/'+ data_list[0])


aa = pd.DataFrame()
for col in ['发动机转速', '油泵转速', '泵送压力', '液压油温', '分配压力',]:
    aa[col+'_peak'] = [signal.find_peaks_cwt(df[col], np.arange(1,2)).shape[0]]
    
for col in ['发动机转速', '油泵转速', '泵送压力',  '分配压力',]: 
    data_1=df
    aa[col+'_error'] = [data_1[col][  (data_1[col] >np.mean(data_1[col])+2 *np.std(data_1[col]))  |  
           (data_1[col] <np.mean(data_1[col])-2 *np.std(data_1[col]))].shape[0]]


    
aa['sample_file_name'] = data_list[0]
aa.to_csv('mydata/data_test_peak.csv', index=False)
print(aa)

for i in tqdm(range(1, len(data_list))):
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv('new_data/test/' + data_list[i])
        

        aa = pd.DataFrame()
        for col in ['发动机转速', '油泵转速', '泵送压力', '液压油温', '分配压力',]:
            aa[col+'_peak'] = [signal.find_peaks_cwt(df[col], np.arange(1,2)).shape[0]]
            
        for col in ['发动机转速', '油泵转速', '泵送压力',  '分配压力',]: 
            data_1=df
            aa[col+'_error'] = [data_1[col][  (data_1[col] >np.mean(data_1[col])+2 *np.std(data_1[col]))  |  
                   (data_1[col] <np.mean(data_1[col])-2 *np.std(data_1[col]))].shape[0]]

        aa['sample_file_name'] = data_list[i]
        aa.to_csv('mydata/data_test_peak.csv', index=False, header=False, mode='a+')
    
    else:
        continue



# In[ ]:


import pandas as pd
import os
from tqdm import *
# -*- coding: utf-8 -*-
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal
data_list = os.listdir('new_data/train/')
df = pd.read_csv('new_data/train/'+ data_list[0])


aa = pd.DataFrame()
for col in ['发动机转速', '油泵转速', '泵送压力', '液压油温', '分配压力',]:
    aa[col+'_peak'] = [signal.find_peaks_cwt(df[col], np.arange(1,2)).shape[0]]
    
for col in ['发动机转速', '油泵转速', '泵送压力',  '分配压力',]: 
    data_1=df
    aa[col+'_error'] = [data_1[col][  (data_1[col] >np.mean(data_1[col])+2 *np.std(data_1[col]))  |  
           (data_1[col] <np.mean(data_1[col])-2 *np.std(data_1[col]))].shape[0]]


#     aa['sample_file_name'] = data_list[i]
    
aa['sample_file_name'] = data_list[0]
aa.to_csv('mydata/data_train_peak.csv', index=False)


for i in tqdm(range(1, len(data_list))):
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv('new_data/train/' + data_list[i])
        

        aa = pd.DataFrame()
        for col in ['发动机转速', '油泵转速', '泵送压力', '液压油温', '分配压力',]:
            aa[col+'_peak'] = [signal.find_peaks_cwt(df[col], np.arange(1,2)).shape[0]]
            
        for col in ['发动机转速', '油泵转速', '泵送压力',  '分配压力',]: 
            data_1=df
            aa[col+'_error'] = [data_1[col][  (data_1[col] >np.mean(data_1[col])+2 *np.std(data_1[col]))  |  
                   (data_1[col] <np.mean(data_1[col])-2 *np.std(data_1[col]))].shape[0]]

        aa['sample_file_name'] = data_list[i]
        aa.to_csv('mydata/data_train_peak.csv', index=False, header=False, mode='a+')
    
    else:
        continue


