import numpy as numP
from decimal import Decimal
from PIL import Image
import glob
import xlrd
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from numpy.linalg import inv
import gc
import random
import math

train_label=[]
train_iamge=[]
softmax_actual_y = [0]*10
bias1 = [random.uniform(0, 1)]*10
bias2 = [random.uniform(0, 1)]*10
weight_nodes    = numP.random.rand(10,784)
weight_hidden_nodes    = numP.random.rand(10,10)
tData = pd.read_excel(r"C:\Users\sheng\Desktop\BigData\icollege\mnist_test.xlsx", sheet_name='mnist_test')
row_N = tData.shape[0]
col_N = tData.shape[1]
print("Column headings:",tData.columns)

for i in range (10):
    x=[]
    y=[]
    temp=tData.iloc[i,0]
    y.append(temp)
    for j in range (784):
        temp=tData.iloc[i,j+1]
        x.append(temp)
    train_label.append(y)
    train_iamge.append(x)
    
train_label = numP.matrix(train_label,dtype='float64')
train_iamge = numP.matrix(train_iamge,dtype='float64') / 255
##print(train_label.shape,train_label[0][0],train_iamge.shape) //(10, 1) [[7.]] (10, 784)
##print(train_iamge[0][0,0:27])
##for i in range(27):
##    if(i!=0):
##        print(train_iamge[0][0,(28*i):(28*(i+1)-1)])
##
weight_nodes    = numP.matrix(weight_nodes,dtype='float64')
weight_hidden_nodes     = numP.matrix(weight_hidden_nodes,dtype='float64')
bia_hidden      = numP.matrix(bias1,dtype='float64')
bia_softmax_op  = numP.matrix(bias2,dtype='float64')

## feed forward & transformation
y_hidden_node = []
y_output_node = []

for j in range(10):
    temp=[]
    for i in range(10):                                                  #
        temp.append(1/(1+math.exp(-1*(train_iamge[j] * (weight_nodes[i].T)+ bia_hidden[0][0,i] ))))
    y_hidden_node.append(temp)
y_hidden_node = numP.matrix(y_hidden_node,dtype='float64')
print("hidden layer y = \n",y_hidden_node,y_hidden_node.shape)

for i in range(10):
    y_output_node.append(1/(1+math.exp(-1*(y_hidden_node[0] * (weight_hidden_nodes[i].T) + bia_softmax_op[0][0,i]))))
    if(train_label[0][0,0]==i):
        softmax_actual_y[i-1] = 1
    else:
        softmax_actual_y[i-1] = 0        
y_output_node = numP.matrix(y_output_node,dtype='float64')
softmax_actual_y = numP.matrix(softmax_actual_y,dtype='float64')
print("predict output layer y = ",y_output_node)
print("actural output layer y = ",train_label[0][0,0]," @ ",softmax_actual_y)

## Total Error of output layer:
total_error = 0
for i in range (10):
    total_error += 0.5 * (math.pow((softmax_actual_y[0][0,i]-y_output_node[0][0,i]),2))
print("Mean squa error = ",total_error)
## backpropogation and min error rate.
##hidden layer nood weight update
update_weight_hidden_layer = []
learning_rate = 0.25
for i in range(10):
    new_weight=[]
    for j in range(10):
        new_weight.append(weight_hidden_nodes[i].T[j][0,0]-learning_rate*(y_hidden_node[0][0,j]*y_output_node[0][0,i]*(1-y_output_node[0][0,i])*(y_output_node[0][0,i]-softmax_actual_y[0][0,i])))
    update_weight_hidden_layer.append(new_weight)
update_weight_hidden_layer = numP.matrix(update_weight_hidden_layer,dtype='float64')
print("hidden_node_weight= \n",weight_hidden_nodes)
print("Update_hidden_node_weight= \n",update_weight_hidden_layer)

update_weight_input_layer = []
learning_rate = 0.45
for i in range(10):
    new_weight=[]
    for j in range(784):
        new_weight.append(weight_hidden_nodes[i].T[j][0,0]-learning_rate*(y_hidden_node[0][0,j]*y_output_node[0][0,i]*(1-y_output_node[0][0,i])*(y_output_node[0][0,i]-softmax_actual_y[0][0,i])))
    update_weight_input_layer.append(new_weight)
update_weight_input_layer = numP.matrix(update_weight_input_layer,dtype='float64')
print("weight_nodes_input_layer= \n",weight_nodes)
print("update_weight_input_layer= \n",update_weight_input_layer)

