
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame,Series
import random
import numpy as np


# -*- coding:utf-8 -*-

def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

    cm = confusion_matrix(y, yp)  # 混淆矩阵

    import matplotlib.pyplot as plt  # 导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar()  # 颜色标签

    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    return plt


inputfile = './data/moment.csv'
data = pd.read_csv(inputfile, encoding='gbk')
# 注意，此处不能用shuffle
sampler = np.random.permutation(len(data))
d = data.take(sampler).values

data_train = d[:int(0.8*len(data)),:] #选取前80%做训练集
data_test = d[int(0.8*len(data)):,:] #选取后20%做测试集
print(data_train.shape)
print(data_test.shape)

# 构建支持向量机模型代码
x_train = data_train[:, 2:]*30 #放大特征
y_train = data_train[:,0].astype(int)
x_test = data_test[:, 2:]*30 #放大特征
y_test = data_test[:,0].astype(int)
print(x_train.shape)
print(x_test.shape)
# 导入模型相关的支持向量机函数  建立并且训练模型
from sklearn import svm
model = svm.SVC()
model.fit(x_train, y_train)
import pickle
pickle.dump(model, open('./save_model/clf.model', 'wb'))

# model = pickle.load(open('svcmodel.model','rb'))
# 导入输出相关的库，生成混淆矩阵
from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train)) # 训练样本的混淆矩阵
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test)) # 测试样本的混淆矩阵
print(cm_train.shape)
print(cm_test.shape)
df1 = DataFrame(cm_train, index = range(1,5), columns=range(1,5))
df2 = DataFrame(cm_test, index = range(1,5), columns=range(1,5))
df1.to_excel('./train_data_xlxs/trainPre.xlsx')
df2.to_excel('./train_data_xlxs/testPre.xlsx')

print(model.score(x_train,y_train)) # 评价模型训练的准确率
print(model.score(x_test,y_test)) # 评价模型测试的准确率


cm_plot(y_train, model.predict(x_train)).show() # cm_plot是自定义的画混淆矩阵的函数
cm_plot(y_test, model.predict(x_test)).show() # cm_plot是自定义的画混淆矩阵的函数








#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#正式开始的数据
inputfile1 = './real_data/real_moment.csv'
data1 = pd.read_csv(inputfile1, encoding='gbk')

sampler = np.random.permutation(len(data1))
d = data1.take(sampler).values

data_train1 = d[:int(0.8*len(data1)),:] #选取前80%做训练集
data_test1 = d[int(0.8*len(data1)):,:] #选取后20%做测试集
print(data_train1.shape)
print(data_test1.shape)



x_train1 = data_train1[:, 2:] * 30 #放大特征
y_train1 = data_train1[:, 0].astype(int)
x_test1 = data_test1[:, 2:] * 30 #放大特征
y_test1 = data_test1[:, 0].astype(int)
print(x_train1.shape)
print(x_test1.shape)


cm_train1 = metrics.confusion_matrix(y_train1, model.predict(x_train1))
# df3 = DataFrame(cm_train1, index = range(1, 5), columns=range(1, 5))
# df3.to_excel('./real_data_xlxs/realPreTrain.xlsx')
# print(model.score(x_train1, y_train1)) # 评价模型测试的准确率
cm_plot(y_train1, model.predict(x_train1)).show() # cm_plot是自定义的画混淆矩阵的函数

cm_test1 = metrics.confusion_matrix(y_test1, model.predict(x_test1))
# df4 = DataFrame(cm_test1, index = range(1, 5), columns=range(1, 5))
# df4.to_excel('./real_data_xlxs/realPreTest.xlsx')
# print(model.score(x_test1, y_test1))
cm_plot(y_test1, model.predict(x_test1)).show()

print(model.score(x_train1, y_train1)) # 评价模型训练的准确率
print(model.score(x_test1, y_test1)) # 评价模型测试的准确率