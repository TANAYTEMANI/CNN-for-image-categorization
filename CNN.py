#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Convolutional Neural Network


# In[2]:


#importing libraries

import keras
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


(x_train,y_train),(x_test,y_test)= mnist.load_data()


# In[4]:


x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(x_test.shape[0],28,28)
input_shape=(28,28,1)


# In[5]:


y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')


# In[6]:


for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(x_train[i],cmap=plt.get_cmap('gray'))
    plt.show()


# In[7]:


x_train /= 255
x_test /= 255
batch_size = 64
num_classes = 10
epochs = 10

def build_model(optimizer):
    model = Sequential()

    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer=optimizer, metrics = ['accuracy'])
    model.summary()
    return model
              
optimizers = ['Adadelta','Adagrad','Adam','RMSprop','SGD']

#for i in optimizers: 

model = build_model('Adam')
hist=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
            


# In[8]:


y1 = hist.history['loss']
plt.plot(y1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[9]:


y2 = hist.history['accuracy']
plt.plot(y2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# In[10]:


keras.models.save_model(model,"mnist.h5",save_format="h5")


# In[11]:


from tensorflow.keras.models import load_model


# In[12]:


def predict_img(model,img):
    img=cv2.resize(img,(28,28))
    # img = img.astype('float32') / 255
    img=np.reshape(img,(1,28,28))
    pred=model.predict(img)
    print(pred)
    print("Class",np.argmax(pred))


# In[13]:


m = load_model('mnist.h5')
predict_img(m,x_train[0])


# In[14]:


plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))


# In[15]:


from sklearn.metrics import accuracy_score


# In[16]:


def predict_all_img(model,img,y):
    img=np.reshape(img,(len(img),28,28))
    pred=model.predict(img)
    pred=np.argmax(pred,axis=1)
    y_true=np.argmax(y,axis=1)
    acc=accuracy_score(y_true,pred)
    return acc


# In[17]:


print("Accuracy on Training data:",predict_all_img(model,x_train,y_train))


# In[18]:


print("Accuracy on Testing data:",predict_all_img(model,x_test,y_test))

