import copy

import numpy as np
import os
from random import random
from pyrsgis import raster
import copy


import tensorflow as tf
from pyrsgis.ml import imageChipsFromArray

# Load the saved model
model = tf.keras.models.load_model('/Users/jiangzongqing/BNU/GraduationThesis/JZQ_CNN_Buildup/CNN_Builtup_PScore0.840_RScore0.981_FScore0.905.h5')

# Load a new multispectral image
# raster = rasterio.open()
ds, featuresHyderabad = raster.read('/Users/jiangzongqing/BNU/GraduationThesis/JZQ_CNN_Buildup/GF1C_PMS_E113.7_N22.4_20210112_L1A1021696543-fusion_BNU_ZH.tif')

# Generate image chips from array
""" Update: 29 May 2021
Note that this time we are generating image chips from array
because we needed the datasource object (ds) to export the TIF file.
And since we are reading the TIF file anyway, why not use the array directly.
"""

new_features = imageChipsFromArray(featuresHyderabad, x_size=7, y_size=7)

print('Shape of the new features', new_features.shape)

# Predict new data and export the probability raster
newPredicted = model.predict(new_features)
#print("Shape of the predicted labels: ", newPredicted.shape)
newPredicted = newPredicted[:,1]

prediction = np.reshape(newPredicted, (ds.RasterYSize, ds.RasterXSize))

outFile = '20220303_BNU_BuiltupCNN_predicted_7by7.tif'
raster.export(prediction, ds, filename=outFile, dtype='float')

from tensorflow import keras
from keras.layers import Conv2D #构造空间卷积层（二维）和池化层
from keras.layers import Dropout #梯度下降，随机丢弃神经网络单元，防止过拟合
from keras.layers import Flatten #将多维输入标准化为一维，使卷积层过度到全连接层不影响batch
from keras.layers import Dense #全连接层，其逻辑为带有权重weight和偏置bias的线性函数
model = keras.Sequential() #顺序模型
# 每一个卷积核扫描过图片都会形成一个特征层，即32个卷积核扫描过后形成32个维度的特征层，新层是上一层的2倍
model.add(Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=(7, 7, 4)))
model.add(Dropout(0.25))
#池化层 减少参数
model.add(Conv2D(48, kernel_size=1, padding='valid', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# 运行model
model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop',metrics=['accuracy'])
model.fit(train_x, train_y)