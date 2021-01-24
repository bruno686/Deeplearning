import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, concatenate
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.backend as K
import tensorflow as tf

# 导入依赖库
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Add, Dense ,Flatten
from keras.models import Model
import numpy as np
from keras.layers import Embedding, LayerNormalization, Dropout, Conv1D
import tensorflow as tf
from keras import regularizers

import warnings
warnings.filterwarnings('ignore')

res_data_x = pd.read_csv('./csv//Indian_pines_corrected.csv')
res_data_y = pd.read_csv('./csv//Indian_pines_gt.csv')

#pca降维
res_X = res_data_x
pca = PCA(n_components=0.9999)
pca.fit(res_X)
res_newX=pca.fit_transform(res_X)
res_newX = pd.DataFrame(res_newX)
res_newdata=pd.concat([res_newX,res_data_y],axis=1)
res_newdata.columns = [np.arange(0,102)]
res_newdata = res_newdata[~((res_newdata[101,] == 0))]
res_newdata[101,] = res_newdata[101,]-1

#数据集划分
res_data = np.array(res_newdata)
res_train_data = res_data[:,:101]
res_train_target = res_data[:,101:]
res_train_X,res_test_X,res_train_y,res_test_y = train_test_split(res_train_data,res_train_target,test_size=0.05,random_state=5)

data_x = pd.read_csv('./csv//Indian_pines_corrected.csv')
data_y = pd.read_csv('./csv//Indian_pines_gt.csv')

newdata = pd.concat([data_x,data_y],axis=1)
print(newdata.shape)
newdata.columns = [np.arange(0,201)]
newdata = newdata[~((newdata[200,] == 0))]
newdata[200,] = newdata[200,]-1
data = np.array(newdata)
print(data.shape)
train_data = data[:,:200]
train_target = data[:,200:]
train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.05,random_state=5)

X_train = np.array(train_X)
X_test = np.array(test_X)
Y_test = keras.utils.to_categorical(test_y, num_classes=16)
Y_train = keras.utils.to_categorical(train_y, num_classes=16)
# X_train = X_train.reshape(9736,101)
# X_test = X_test.reshape(513,101)

res_X_train = np.array(res_train_X)
res_X_test = np.array(res_test_X)
res_Y_test = keras.utils.to_categorical(res_test_y, num_classes=16)
res_Y_train = keras.utils.to_categorical(res_train_y, num_classes=16)
res_X_train = res_X_train.reshape(9736,1,1,101)
res_X_test = res_X_test.reshape(513,1,1,101)

ROWS = 1
COLS = 1
CHANNELS = 101
CLASSES = 16


def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

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
positional_encoding = get_positional_encoding(max_seq_len=200, embed_dim=24)

def self_attention(X,X_Position_Embedding):
    W_Q=np.random.randn(X.shape[2],X.shape[2])
    W_K=np.random.randn(X.shape[2],X.shape[2])
    W_V=np.random.randn(X.shape[2],X.shape[2])
    Q = X@W_Q
    K = X@W_K
    V = X@W_V
    K = tf.transpose(K,perm=[0, 2, 1])
    X = (Q@K@V)/X_Position_Embedding.shape[2]
    X = Dense(24, activation='softmax')(X)
    return X

def transformer(X,positional_encoding):
    X_shortcut1 = X
    X_selfatteention = self_attention(X,positional_encoding)
    X_Drop1 = Dropout(0.5)(X_selfatteention)
    X_add1 = Add()([X_Drop1, X_shortcut1])

    X_Laynorm1 = LayerNormalization()(X_add1)
    X_shortcut2 = X_Laynorm1
    X_feed1 = Dense(24,activation='relu',kernel_regularizer=regularizers.l2(0.1))(X_Laynorm1)
    X_feed2 = Dense(24,activation='relu',kernel_regularizer=regularizers.l2(0.1))(X_feed1)
    x_Drop2 = Dropout(0.5)(X_feed2)
    X_add2 = Add()([x_Drop2, X_shortcut2])

    X_Laynorm2 = LayerNormalization()(X_add2)
    return X_Laynorm2

input_shape = (200,)
X_input = Input(input_shape)
X_Flat = Flatten()(X_input)
X_Embedding = Embedding(input_dim=20000, output_dim=24)(X_Flat)
X_Position_Embedding = positional_encoding+X_Embedding
X_Dense1 = Dense(24, activation='relu',kernel_regularizer=regularizers.l2(0.1))(X_Position_Embedding)
X_transformer1 = transformer(X_Dense1,X_Position_Embedding)
X_conv1D = Conv1D(24,1,strides=1,activation='relu')(X_transformer1)
# X_Dense = Dense(24, activation='relu',kernel_regularizer=regularizers.l2(0.1))(X_transformer1)
# X_transformer2 = transformer(X_Dense)
# X_transformer3 = transformer(X_transformer2)
# X_transformer4 = transformer(X_transformer3)
output = Dense(24,activation='relu',kernel_regularizer=regularizers.l2(0.1))(X_conv1D)
output = Flatten()(output)
output = Dense(16,activation='softmax')(output)

res_input_shape = (1, 1, 101)
classes = 16
# Define the input as a tensor with shape input_shape
res_X_input = Input(res_input_shape)

# Zero-Padding
res_X = ZeroPadding2D((3, 3))(res_X_input)

# Stage 1
res_X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(res_X)
res_X = BatchNormalization(axis=3, name='bn_conv1')(res_X)
res_X = Activation('relu')(res_X)
res_X = MaxPooling2D((1, 1), strides=(2, 2))(res_X)

# Stage 2
res_X = convolutional_block(res_X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
res_X = identity_block(res_X, 3, [64, 64, 256], stage=2, block='b')
res_X = identity_block(res_X, 3, [64, 64, 256], stage=2, block='c')

# Stage 3
res_X = convolutional_block(res_X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
res_X = identity_block(res_X, 3, [128, 128, 512], stage=3, block='b')
res_X = identity_block(res_X, 3, [128, 128, 512], stage=3, block='c')
res_X = identity_block(res_X, 3, [128, 128, 512], stage=3, block='d')

# Stage 4
res_X = convolutional_block(res_X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
res_X = identity_block(res_X, 3, [256, 256, 1024], stage=4, block='b')
res_X = identity_block(res_X, 3, [256, 256, 1024], stage=4, block='c')
res_X = identity_block(res_X, 3, [256, 256, 1024], stage=4, block='d')
res_X = identity_block(res_X, 3, [256, 256, 1024], stage=4, block='e')
res_X = identity_block(res_X, 3, [256, 256, 1024], stage=4, block='f')

# Stage 5
res_X = convolutional_block(res_X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
res_X = identity_block(res_X, 3, [512, 512, 2048], stage=5, block='b')
res_X = identity_block(res_X, 3, [512, 512, 2048], stage=5, block='c')

# AVGPOOL.
res_X = AveragePooling2D((1, 1), name='avg_pool')(res_X)

# output layer
res_X = Flatten()(res_X)
res_X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(res_X)
output = concatenate([res_X, output])

# Create model
model = Model(inputs = [res_X_input,X_input], outputs = output, name='resnet')
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


batch = 200
model.fit([res_X_train,X_train], Y_train, epochs=60, batch_size=batch)
scores = model.evaluate([res_X_test,X_test], Y_test)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))