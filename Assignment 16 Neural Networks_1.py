#!/usr/bin/env python
# coding: utf-8

# PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS

# In[1]:


import pandas as pd
import numpy as np
from keras.models import Sequential
import keras


# In[2]:


data=pd.read_csv('Forestfires.csv')


# In[3]:


data


# In[4]:


from sklearn.preprocessing import LabelEncoder
l_en=LabelEncoder()
data['month']=l_en.fit_transform(data['month'])
data['day']=l_en.fit_transform(data['day'])


# In[5]:


data1=data.iloc[:,0:11]
data1


# In[6]:


x=data1.drop('area',axis=1)
y=data1['area']


# In[7]:


from sklearn.preprocessing import StandardScaler
a=StandardScaler()
x_standardized=a.fit_transform(x)


# In[8]:


pd.DataFrame(x_standardized).describe()


# # Tuning of hyperparameters: BatchSize and Epochs

# In[9]:


from sklearn.model_selection import GridSearchCV,KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam


# In[10]:


def create_model():
    model=Sequential()
    model.add(Dense(15,input_dim=10,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    adam=Adam(lr=0.01)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# In[11]:


model=KerasClassifier(build_fn=create_model,verbose=0)
batch_size=[20,50,77]
epochs=[25,75,100]
param_grid=dict(batch_size=batch_size,epochs=epochs)
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# # Tuning of hyperparameters: LearningRate and DropOut

# In[12]:


from keras.layers import Dropout
def create_model(learning_rate,dropout_rate):
    model=Sequential()
    model.add(Dense(10,input_dim=10,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(6,input_dim=10,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[13]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=20,epochs=25)
learning_rate=[0.1,0.01,0.001]
dropout_rate=[0.2,0.02,0.002]
param_grids=dict(learning_rate=learning_rate,dropout_rate=dropout_rate)
grid=GridSearchCV(estimator=model,param_grid=param_grids,cv=KFold(),verbose=10)


# In[14]:


grid_result=grid.fit(x_standardized,y)


# In[15]:


print('Best : {}, Using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Activation function and Kernel initializer

# In[17]:


def create_model(activation_function,init):
    model=Sequential()
    model.add(Dense(9,input_dim=10,kernel_initializer=init,activation=activation_function))
    model.add(Dropout(0.2))
    model.add(Dense(5,input_dim=10,kernel_initializer=init,activation=activation_function))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=0.1)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[18]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=20,epochs=25)
activation_function=['softmax','relu','tanh','linear']
init=['uniform','normal','zero']
param_grid=dict(activation_function=activation_function,init=init)
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# # Number of neurons in activation layer

# In[21]:


def create_model(neuron1,neuron2):
    model=Sequential()
    model.add(Dense(neuron1,input_dim=10,kernel_initializer='normal',activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(neuron2,input_dim=10,kernel_initializer='normal',activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=0.1)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[22]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=20,epochs=25)
neuron1=[30,20,10]
neuron2=[10,15,20]
param_grid=dict(neuron1=neuron1,neuron2=neuron2)
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# In[23]:


print('Best : {}, Using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Train model with optimum values of hyperparameter

# In[24]:


model=Sequential()
model.add(Dense(30,input_dim=10,activation='linear',kernel_initializer='normal'))
model.add(Dropout(0.2))
model.add(Dense(10,input_dim=10,kernel_initializer='normal',activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

adam=Adam(lr=0.1)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[25]:


model.fit(x,y, verbose=0, batch_size=20, epochs=25)


# In[27]:


y_predict=model.predict(x_standardized)
cutoff = 0.7
y_predict_classes = np.zeros_like(y_predict)
y_predict_classes[y_predict > cutoff] = 1


# In[28]:


y_classes = np.zeros_like(y_predict)
y_classes[y > cutoff] = 1


# In[30]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_classes,y_predict_classes))

