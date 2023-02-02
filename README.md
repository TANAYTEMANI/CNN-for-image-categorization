# CNN-for-image-categorization

CNN stands for Convolutional Neural Network, which is a type of deep learning neural network used for image recognition and processing. It uses a process called convolution, where filters are applied to input data to extract meaningful features, which are then passed through multiple layers to produce an output. CNNs have achieved state-of-the-art results on tasks such as image classification, object detection, and semantic segmentation.The network is composed of multiple layers, including convolutional layers, activation layers, pooling layers, and fully connected layers.

## Input
![image](https://user-images.githubusercontent.com/82306595/216334559-4ae6089a-3802-4da6-8f17-ba49cb8222fc.png)

### Importing libraries
```javascript I'm A tab
import keras
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
```

### Load Data
```javascript I'm A tab
(x_train,y_train),(x_test,y_test)= mnist.load_data()
```

### Reshape Image
```javascript I'm A tab
x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(x_test.shape[0],28,28)
input_shape=(28,28,1)
```

### Categorizing Features
```javascript I'm A tab
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
```

```javascript I'm A tab
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(x_train[i],cmap=plt.get_cmap('gray'))
    plt.show()
```

### CNN Model
```javascript I'm A tab
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
```

### Loss vs Epoch
```javascript I'm A tab
y1 = hist.history['loss']
plt.plot(y1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```
![image](https://user-images.githubusercontent.com/82306595/216337877-7c286c52-eb5e-4fd0-aa54-bded092fcc38.png)


### Accuracy vs Epoch
```javascript I'm A tab
y2 = hist.history['accuracy']
plt.plot(y2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
```
![image](https://user-images.githubusercontent.com/82306595/216337955-a1b59d72-9795-4611-af7a-f807fb90a4e2.png)


### Prediction
```javascript I'm A tab
keras.models.save_model(model,"mnist.h5",save_format="h5")
from tensorflow.keras.models import load_model
```

```javascript I'm A tab
def predict_img(model,img):
    img=cv2.resize(img,(28,28))
    # img = img.astype('float32') / 255
    img=np.reshape(img,(1,28,28))
    pred=model.predict(img)
    print(pred)
    print("Class",np.argmax(pred))
```


### Predicted Output
```javascript I'm A tab
m = load_model('mnist.h5')
predict_img(m,x_train[0])
```

```javascript I'm A tab
plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))
```

### Accuracy
```javascript I'm A tab
from sklearn.metrics import accuracy_score
```
```javascript I'm A tab
def predict_all_img(model,img,y):
    img=np.reshape(img,(len(img),28,28))
    pred=model.predict(img)
    pred=np.argmax(pred,axis=1)
    y_true=np.argmax(y,axis=1)
    acc=accuracy_score(y_true,pred)
    return acc
```
```javascript I'm A tab
print("Accuracy on Training data:",predict_all_img(model,x_train,y_train))
```
```javascript I'm A tab
print("Accuracy on Testing data:",predict_all_img(model,x_test,y_test))
```
