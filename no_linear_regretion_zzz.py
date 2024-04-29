#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:31:37 2023 UTEC

@author: Cristian López Del Alamo
"""


import numpy as np
import matplotlib.pyplot as ptl 

class LinearRegresion:
    def __init__(self, grado):
        self.m_W = np.random.rand(grado)
        self.m_b = np.random.random()
        self.grado = grado
      
        
    def H(self,X):
        return np.dot(X,self.m_W)
    # + self.m_b

    def predic(self,x):
       potencias = np.arange(self.grado)
       x = np.power.outer(x, potencias) 
       return np.dot(x,self.m_W)
       

    
    def Loss(self,X,Y,lambda_):
        y_pred = self.H(X)
        return (np.linalg.norm((Y   - y_pred))**2)/(2*len(Y)), y_pred  +  lambda_*np.linalg.norm(self.m_W)

    def dL(self,X,Y, Y_pre,lambda_):
        dw =  np.matmul(Y - Y_pre,-X)/len(Y) + 2*lambda_*self.m_W
        db =  np.sum((Y - Y_pre)*(-1))/len(Y)
        return dw,db 

    def change_params(self, dw,db, alpha):
        self.m_W = self.m_W - alpha*dw 
        # self.m_b = self.m_b - alpha*db

    def train(self, X, Y, alpha,epochs,lambda_):
        error_list = []
        time_stamp = []
        
        potencias = np.arange(self.grado)
        X = np.power.outer(X, potencias)
        
        for i in range(epochs):
            loss, y_pred = self.Loss(X,Y,lambda_)
            time_stamp.append(i)
            error_list.append(loss)
            dw,db = self.dL(X,Y,y_pred,lambda_)
            self.change_params(dw,db,alpha)
           
            if(i%1000==0):
               # self.plot_error(time_stamp, error_list)
               print("error de pérdida : " + str(loss))
               #LR.plot_line(x, LR.predic(x))
        return time_stamp, error_list
       

    def plot_error(self, time, loss):
       ptl.plot(time, loss)
       ptl.show()
  
    def plot_line(self,x,y_pre):
       ptl.plot(x, y_pre,'.')
       ptl.plot(x, y_pre,'.')
      






path = 'C:\\Users\\Cristian\\' 

data_train = np.genfromtxt(path + 'train.csv',delimiter = ',')
data_test = np.genfromtxt(path + 'test.csv',delimiter = ',')

n = 20

#x = np.linspace(0, 2*np.pi, n)
#y = np.sin(x) + np.random.normal(0,100)

x = data_train[0]

y = data_train[1]

x_test = data_test

x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

x_test = (x_test - np.min(y)) / (np.max(y) - np.min(y))


# rst = input("Quiere ver los resultados : (Y/Ny)")
LR = LinearRegresion(15)
time, loss = LR.train(x,y,0.7,100000,0.0001)
LR.plot_error(time, loss)
# # rst = input("Quiere la aproxmación de la función : (Y/Ny)")
y_aprox = LR.predic(x_test)
print(y_aprox.shape)
print(x_test.shape)

ptl.plot(x_test, y_aprox,'*')
ptl.plot(x, y,'*')




