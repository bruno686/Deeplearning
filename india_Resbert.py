import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import load_model

res_data_x = pd.read_csv('./csv//Indian_pines_corrected.csv')  #初始加载X(21025, 200)
res_data_y = pd.read_csv('./csv//Indian_pines_gt.csv')         #初始加载Y(21025, 1)
res_first_concat = pd.concat([res_data_x,res_data_y],axis=1)   #将初始X和初始Y拼接(21025, 201)
#pca降维
# res_X = res_data_x
pca = PCA(n_components=0.9999)
pca.fit(res_data_x)
res_pca_X=pca.fit_transform(res_data_x)                             #对Xpca降维处理X(21025, 201)->(21025, 101)
#适合于产出图像
res_image_X = res_pca_X.reshape(21025,1,1,101)

data_x = pd.read_csv('./csv//Indian_pines_corrected.csv')
data_y = pd.read_csv('./csv//Indian_pines_gt.csv')
newdata_X = np.array(data_x)
NEWdata = pd.concat([data_x,data_y],axis=1)
NEWdata = np.array(NEWdata)

Resbert9142 = load_model('/home/bruno/文档/2020GIT/Resbert9142.h5')
ypred = Resbert9142.predict([res_image_X,newdata_X])

clmap = np.array(ypred).reshape(145, 145,16).astype('float')
clmap = np.argmax(clmap,axis=2)
plt.figure(figsize=(10, 8))
plt.imshow(clmap, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('India Resbert.png')
plt.xlabel("India Resbert.png")
plt.show()