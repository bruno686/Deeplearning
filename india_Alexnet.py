import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

data_x = pd.read_csv('./csv//Indian_pines_corrected.csv')
data_y = pd.read_csv('./csv//Indian_pines_gt.csv')
q = pd.concat([data_x,data_y],axis=1)
#pca降维
X = data_x
pca = PCA(n_components=0.9999)
pca.fit(X)
newX=pca.fit_transform(X)
newdata_X = newX.reshape(21025,101,1)
newX = pd.DataFrame(newX)
newdata=pd.concat([newX,data_y],axis=1)
newdata.columns = [np.arange(0,102)]
newdata = newdata[~((newdata[101,] == 0))]
newdata[101,] = newdata[101,]-1

#数据集划分
data = np.array(newdata)
train_data = data[:,:101]
train_target = data[:,101:]
train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.05,random_state=5)

X_train = np.array(train_X)
X_test = np.array(test_X)
Y_test = keras.utils.to_categorical(test_y, num_classes=16)
Y_train = keras.utils.to_categorical(train_y, num_classes=16)
X_train = X_train.reshape(9736,101,1)
X_test = X_test.reshape(513,101,1)


Alexnet_input = Input(shape=(101,1))

Conv1 = Conv1D(32,4,activation='relu')(Alexnet_input)
Batchnormal1 = BatchNormalization()(Conv1)
Maxpool1 = MaxPooling1D(2)(Batchnormal1)

Conv2 = Conv1D(32,4,activation='relu')(Maxpool1)
Batchnormal2 = BatchNormalization()(Conv2)
Maxpool2 = MaxPooling1D(2)(Batchnormal2)

Conv3 = Conv1D(192,3,strides=1,activation='relu')(Maxpool2)
Conv4 = Conv1D(192,3,strides=1,activation='relu')(Conv3)
Conv5 = Conv1D(192,3,strides=1,activation='relu')(Conv4)
Maxpool3 = MaxPooling1D(2)(Conv5)
Flat = Flatten()(Maxpool3)
Den1 = Dense(256, activation='relu')(Flat)
Drop1 = Dropout(0.5)(Den1)

Den2 = Dense(256, activation='relu')(Drop1)
Drop2 = Dropout(0.5)(Den2)
output = Dense(16, activation='softmax')(Drop2)

model = Model(inputs=Alexnet_input,outputs=output)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch =1000
model.fit(X_train, Y_train, epochs=20, batch_size=batch)
scores = model.evaluate(X_test, Y_test)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

ypred = model.predict(newdata_X)

clmap = np.array(ypred).reshape(145, 145,16).astype('float')
clmap = np.argmax(clmap,axis=2)
plt.figure(figsize=(10, 8))
plt.imshow(clmap, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('India Alexnet.png')
plt.xlabel("India Alexnet")
plt.show()