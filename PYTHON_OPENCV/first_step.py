# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 20:16:15 2018

@author: Administrator
"""

#导入库文件
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd



#计算颜色矩特征模型
def img2vector(filename):


    returnvect = np.zeros((1, 9))
    #一个1*9的二维数组


    fr = mpimg.imread(filename)
    #用matplotlib读取图片文件


    l_max = fr.shape[0]//2+50   #读取矩阵的第一个维度，然后除以二后向下取整，再加50
    l_min = fr.shape[0]//2-50
    w_max = fr.shape[1]//2+50
    w_min = fr.shape[1]//2-50
    water = fr[l_min:l_max, w_min:w_max, :].reshape(1, 10000, 3)#重塑为一个三维矩阵，1*10000*3


    for i in range(3):
        this = water[:, :, i]/255
        print(this)
        returnvect[0, i] = np.mean(this)    #0,1,2存储一阶颜色矩
        returnvect[0, 3+i] = np.sqrt(np.mean(np.square(this-returnvect[0, i])))#3,4,5存储二阶颜色矩
        returnvect[0, 6+i] = np.cbrt(np.mean(np.power(this-returnvect[0, i], 3)))#6,7,8存储三阶颜色矩
        print(returnvect)
    return returnvect





#计算每个图片的特征
trainfilelist = os.listdir('./water_image')#读取目录下文件列表
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
    trainingMat[i, :] = img2vector('./water_image/%s' % filenamestr) #构成数组

#保存
d = np.concatenate((labels.T, train.T, trainingMat), axis=1)#连接数组
dataframe = pd.DataFrame(d, columns=['Water kind','number', 'R_1', 'G_1', 'B_1', 'R_2', 'G_2', 'B_2', 'R_3', 'G_3', 'B_3'])
dataframe.to_csv('./data/moment.csv', encoding='utf-8', index=False)#保存文件


