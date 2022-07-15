import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense

# Prepare data
df = pd.read_excel(r'D:\Dev\0617_P1\EX02.xlsx', 'ex01')
# sys.exit()

df_01 = df[['ex']]
# df_01 = df.iloc[:,[12]]
df_lft = df.drop(['ex'], axis=1)
df_lft = df_lft.drop(['Time'], axis=1)
x = df_lft.to_numpy()
y = df_01.to_numpy()
# x = x[0:2]
# y = y[0:2]

print("x = \n" , x)
print("y = \n" , y)
print(x.shape)
print("x = \n" , x, '\n', x.shape)
print("y = \n" , y, '\n', y.shape)

y_nw = np.eye(15)[y]
y_nw = np.reshape(y_nw, (-1, 15))
print("2Dy = \n" , y_nw, '\n', y_nw.shape)

# sys.exit()

# split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# 모델 정의
dnn = keras.Sequential()
dnn.add(Dense(units=128, activation='relu', input_shape=(13,)))
dnn.add(Dense(units=15, activation='softmax'))

# # 모델 확인
dnn.summary()

# 3. 실행
dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 학습시키기
dnn.fit(x, y_nw, epochs=100)

# 결과 출력
score = dnn.evaluate(x, y_nw, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


