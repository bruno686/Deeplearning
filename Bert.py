# 导入依赖库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.backend as K
import tensorflow as tf

data_x = pd.read_csv('./csv//Indian_pines_corrected.csv')
data_y = pd.read_csv('./csv//Indian_pines_gt.csv')
#pca降维
X = data_x
pca = PCA(n_components=0.9999)
pca.fit(X)
newX=pca.fit_transform(X)
newX = pd.DataFrame(newX)
newdata=pd.concat([newX,data_y],axis=1)
newdata.columns = [np.arange(0,102)]
newdata = abs(newdata)
newdata = newdata[~((newdata[101,] == 0))]
newdata[101,] = newdata[101,]-1

#数据集划分
data = np.array(X)
# data = np.array(newdata)
print(data.shape)
train_data = data[:,:101]
train_target = data[:,101:]
train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.05,random_state=5)


X_train = np.array(train_X)
X_test = np.array(test_X)
Y_test = keras.utils.to_categorical(test_y, num_classes=16)
Y_train = keras.utils.to_categorical(train_y, num_classes=16)
X_train = X_train.reshape(9736,101)
X_test = X_test.reshape(513,101)

ROWS = 1
COLS = 1
CHANNELS = 101
CLASSES = 16

from keras.layers import Embedding,LayerNormalization,Dropout
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def transformer(X):
    X_Drop1 = Dropout(0.5)(X)
    X_Laynorm1 = LayerNormalization()(X_Drop1)
    X_shortcut = X_Laynorm1
    X_feed1 = Dense(50,activation='relu')(X_Laynorm1)
    X_feed2 = Dense(24,activation='relu')(X_feed1)
    X_Feedforward = Add()([X_feed2, X_shortcut])
    X_Drop2 = Dropout(0.5)(X_Feedforward)
    X_Laynorm2 = LayerNormalization()(X_Drop2)
    return X_Laynorm2

def self_attention(X):
    W_Q=np.random.randn(X.shape[2],X.shape[2])
    W_K=np.random.randn(X.shape[2],X.shape[2])
    W_V=np.random.randn(X.shape[2],X.shape[2])
    Q = X@W_Q
    K = X@W_K
    V = X@W_V
    K = np.transpose(K)
    X = (Q@K@V)/X_Position_Embedding.shape[2]
    X = Dense(24, activation='softmax')(X)
    print(X.shape)
    return X

def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    # 归一化, 用位置嵌入的每一行除以它的模长
    # denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    # position_enc = position_enc / (denominator + 1e-8)
    return positional_encoding

positional_encoding = get_positional_encoding(max_seq_len=101, embed_dim=24)
input_shape = (101,)
X_input = Input(input_shape)
X_Flat = Flatten()(X_input)
X_Embedding = Embedding(input_dim=20000, output_dim=24)(X_Flat)
X_Position_Embedding = positional_encoding+X_Embedding
X_transformer1 = transformer(X_Position_Embedding)
X_transformer2 = transformer(X_transformer1)
X_transformer3 = transformer(X_transformer2)
X_transformer4 = transformer(X_transformer3)
output = Dense(24,activation='relu')(X_transformer4)
output = Flatten()(output)
output = Dense(16,activation='softmax')(output)
print(output.shape)


model = Model(inputs=X_input,outputs=output)
print(X_input.shape)
print(output.shape)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch =1000

model.fit(X_train, Y_train, epochs=50, batch_size=batch)
scores = model.evaluate(X_test, Y_test)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

