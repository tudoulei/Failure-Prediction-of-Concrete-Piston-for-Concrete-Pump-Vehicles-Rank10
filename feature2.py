
# 制作第二个特征集



import pandas as pd
import os
from tqdm import *
path="new_data/"
data_list = os.listdir(path+'train/')


file_name='mydata/data_all_n2_d5_train.csv'
df = pd.read_csv(path+'train/'+ data_list[0])

df.drop(df[df['活塞工作时长']>150].index)  ##50
df.drop(df[ (df['发动机转速']>10000) ].index)
df.drop(df[  (df['油泵转速']>15000) ].index)
#df.drop(df[ df['泵送压力']>450 ].index)  ##>0
df.drop(df[ df['排量电流']>50000 ].index)  ##
df['t1']=df['发动机转速']*df['分配压力']  #
df['t2']=df['发动机转速']*df['泵送压力'] #
df['t3']=df['搅拌超压信号']/df['活塞工作时长']  ##
df['t4']=df['分配压力']/df['泵送压力']  ##
df['t5']=df['油泵转速']*df['泵送压力']  ##
df['t6']=df['排量电流'] /df['流量档位']    ###每?
df['t7']=[ 0    if x<85 and x>0  else  1  for x in df['液压油温'] ]   #

df['sample_file_name'] = data_list[0]
df.to_csv(file_name, index=False,encoding='utf-8')

for i in tqdm(range(1, len(data_list))):
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'train/' + data_list[i])

        df.drop(df[df['活塞工作时长']>150].index)  ##50
        df.drop(df[ (df['发动机转速']>10000) ].index)
        df.drop(df[  (df['油泵转速']>15000) ].index)
        #df.drop(df[ df['泵送压力']>450 ].index)  ##>0
        df.drop(df[ df['排量电流']>50000 ].index)  ##由训练数据图表观察到异常      
        df['t1']=df['发动机转速']*df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t2']=df['发动机转速']*df['泵送压力'] #？？
        df['t3']=df['搅拌超压信号']/df['活塞工作时长']  ##出错机率
        df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
        df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
        df['t6']=df['排量电流'] /df['流量档位']    ###每?
        df['t7']=[ 0    if x<85 and x>0  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏
  
        df['sample_file_name'] = data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue


# In[5]:




import pandas as pd
import os
from tqdm import *
path="new_data/"
test_data_list = os.listdir(path+'test/')


file_name='mydata/data_all_n2_d5_test.csv'
df = pd.read_csv(path+'test/'+ test_data_list[0])

df.drop(df[df['活塞工作时长']>150].index)  ##50
df.drop(df[ (df['发动机转速']>10000) ].index)
df.drop(df[  (df['油泵转速']>15000) ].index)
#df.drop(df[ df['泵送压力']>450 ].index)  ##>0
df.drop(df[ df['排量电流']>50000 ].index)  ##由训练数据图表观察到异常      
df['t1']=df['发动机转速']*df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
df['t2']=df['发动机转速']*df['泵送压力'] #？？
df['t3']=df['搅拌超压信号']/df['活塞工作时长']  ##出错机率
df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
df['t6']=df['排量电流'] /df['流量档位']    ###每?
df['t7']=[ 0    if x<85 and x>0  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏

df['sample_file_name'] = data_list[0]
df.to_csv(file_name, index=False,encoding='utf-8')


for i in tqdm(range(len(test_data_list))):
    if test_data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'test/' + test_data_list[i])

        df.drop(df[df['活塞工作时长']>150].index)  ##50
        df.drop(df[ (df['发动机转速']>10000) ].index)
        df.drop(df[  (df['油泵转速']>15000) ].index)
        #df.drop(df[ df['泵送压力']>450 ].index)  ##>0
        df.drop(df[ df['排量电流']>50000 ].index)  ##由训练数据图表观察到异常      
        df['t1']=df['发动机转速']*df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t2']=df['发动机转速']*df['泵送压力'] #？？
        df['t3']=df['搅拌超压信号']/df['活塞工作时长']  ##出错机率
        df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
        df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
        df['t6']=df['排量电流'] /df['流量档位']    ###每?
        df['t7']=[ 0    if x<85 and x>0  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏 
            
        df['sample_file_name'] = test_data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue


# In[7]:



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
#              ,'设备类型'
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

train = pd.read_csv('mydata/data_all_n2_d5_train.csv',dtype=column_types)
test = pd.read_csv('mydata/data_all_n2_d5_test.csv',dtype=column_types)

sample = pd.read_csv('new_data/submit_example.csv')


# In[11]:


train.columns


# In[17]:


train['设备类型'] = datashebei_train
test['设备类型'] = datashebei_test


# In[18]:


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


# In[19]:


def signal_is_1(x):
    return np.sum(x == 1)
def signal_is_0(x):
    return np.sum(x == 0)

from scipy import signal

def peak_num_1(x):
    return len(signal.find_peaks_cwt(x, np.arange(1,50)))

def peak_num_2(x):
    return len(signal.find_peaks_cwt(x, np.arange(1,2)))


# In[20]:


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


# In[21]:


data_train.to_csv('mydata/data_train_kaiyaun.csv')

data_test.to_csv('mydata/data_test_kaiyaun.csv',index=None)

