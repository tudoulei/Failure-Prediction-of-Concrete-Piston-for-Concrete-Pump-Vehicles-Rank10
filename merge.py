import os.path
import pandas as pd



filepath = '复现数据/'
pathDir =  os.listdir(filepath)
data = pd.DataFrame()
i = 0
for allDir in pathDir:
#     print(allDir)
    o=open(filepath+'/'+allDir)
    i+=1
    data[allDir] =  pd.read_csv(o)['label']
    o.close()
    


result=pd.DataFrame()

def mean_data(filepath):
    pathDir =  os.listdir(filepath)
    data = pd.DataFrame()
    i = 0
    for allDir in pathDir:
    #     print(allDir)
        o=open(filepath+'/'+allDir)
        i+=1
        data[allDir] =  pd.read_csv(o)['label']
        o.close()
    return data


result['sample_file_name']= pd.read_csv('new_data/submit_example.csv')['sample_file_name'] 
result['prediction']=list(map(lambda x:1 if x>0.45 else 0,data.mean(axis=1)))

file_name ='最终提交'
try:
    os.mkdir(file_name)
except:
    print('已存在')
result.to_csv(file_name+'/最终提交.csv',index=None)