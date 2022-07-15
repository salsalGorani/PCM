# 참고 : 2. 워드 임베딩(Word Embedding) 항목
# https://wikidocs.net/32105

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import pandas as pd
from tensorflow import keras

from keras.layers import Dense
from keras.layers import Embedding
# from keras.layers import Input, Embedding, concatenate
from sklearn.model_selection import train_test_split

# dict_max = {'age':6, 'gen':2, 'ses':4, 'time':5, 'gold':14}
MAX_VOCA = 17   # 0~16
MAX_GOLD = 3   # 0~13
EMB_OUT_DIMS = 128
EMB_INPUT_LEN = 4

# Prepare data
# df = pd.read_excel(r'D:\Dev\0617_P1\EX02.xlsx')
# df = pd.read_excel('EX02.xlsx')
df = pd.read_excel(r'D:\Dev\0617_P1\EX02.xlsx', 'ex02')
df_01 = df[['wts']]

df_lft = df.drop(['wts'], axis=1)
x = df_lft.to_numpy()
y = df_01.to_numpy()
# x = x[0:2]
# y = y[0:2]

print("x = \n", x)
print("y = \n", y)
print("x = \n", x, '\n', x.shape)
print("y = \n", y, '\n', y.shape)
# sys.exit()

y_nw = np.eye(MAX_GOLD)[y]
y_nw = np.reshape(y_nw, (-1, MAX_GOLD))
print("2Dy = \n", y_nw, '\n', y_nw.shape)

# split data
X_train, X_test, y_train, y_test = train_test_split(x, y_nw, test_size=0.33, shuffle=True)

# 모델 정의
dnn = keras.Sequential()
dnn.add(Embedding(MAX_VOCA, EMB_OUT_DIMS, input_length=EMB_INPUT_LEN))
dnn.add(keras.layers.Flatten())
dnn.add(Dense(units=512, activation='relu'))
dnn.add(Dense(units=128, activation='relu'))
dnn.add(Dense(units=32, activation='relu'))
dnn.add(Dense(units=MAX_GOLD, activation='softmax'))

# # 모델 확인
dnn.summary()

# 3. 실행
dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 학습시키기
dnn.fit(X_train, y_train, epochs=100)

# 결과 출력
score = dnn.evaluate(X_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


