import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn import svm
from sklearn import metrics
import joblib
import warnings
from first_step import img2vector
from sklearn.model_selection import GridSearchCV
import seaborn as sns
warnings.filterwarnings("ignore")#防止标签缺失的警报



#计算每个图片的特征
trainfilelist = os.listdir('./video_cut')#读取目录下文件列表
m = len(trainfilelist)                   #计算文件数目
labels = np.zeros((1, m)) #生成两个196个0的空矩阵
train = np.zeros((1, m))
#trainingMat=[]
#print(trainfilelist)
trainingMat=np.zeros((m, 9)) #m行9列的0空矩阵

for i in range(m):
    filenamestr = trainfilelist[i]      #获取当前文件名，例1_1.jpg
    filestr = filenamestr.split('.')[0]  #按照.划分，取前一部分
    classnumstr = int(filestr.split('_')[0])#按照_划分，后一部分为该类图片中的序列
    picture_num = int(filestr.split('_')[1])
    labels[0, i] = classnumstr               #前一部分为该图片的标签
    train[0, i] = picture_num
    trainingMat[i, :] = img2vector('./video_cut/%s' % filenamestr) #构成数组

#保存
d = np.concatenate((labels.T, train.T, trainingMat), axis=1)#连接数组
dataframe = pd.DataFrame(d, columns=['Water kind', 'number', 'R_1', 'G_1', 'B_1', 'R_2', 'G_2', 'B_2', 'R_3', 'G_3', 'B_3'])
dataframe.to_csv('./real_data/real_moment.csv', encoding='utf-8', index=False)#保存文件
