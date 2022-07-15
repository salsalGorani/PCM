import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

# Prepare data
df = pd.read_excel(r'D:\Dev\0617_P1\EX02.xlsx', 'ex02')

# Embedding
max_gold = 3
max_voca = 17
EMB_OUT_DIMS = 128
EMB_INPUT_LEN = 4

# df_01 = df.iloc[:,[12]]
df_lft = df.drop(['ex'], axis=1)
df_lft = df_lft.drop(['Time'], axis=1)
df_lft = df_lft.drop(['wts'], axis=1)
x = df_lft.to_numpy()
print(x, '\n', x.shape)

# y-값('wts' column) numpy화, one-hot encoding, reshaping
df_01 = df[['wts']]
y = df_01.to_numpy()
y_nw = np.eye(max_gold)[y]
y_nw = np.reshape(y_nw, (-1, 3))
print("2Dy = \n" , y_nw, '\n', y_nw.shape)

# 모델 정의
dnn = keras.Sequential()
dnn.add(Embedding(MAX_VOCA, EMB_OUT_DIMS, input_length=EMB_INPUT_LEN))
dnn.add(keras.layers.Flatten())
dnn.add(Dense(units=128, activation='relu', input_shape=(12,)))
dnn.add(Dense(units=3, activation='softmax'))

# split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# # 모델 확인
dnn.summary()
# sys.exit()

# 3. 실행
dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 학습시키기
dnn.fit(x, y_nw, epochs=100)

# 결과 출력
score = dnn.evaluate(x, y_nw, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


