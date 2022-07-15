import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Prepare data
df = pd.read_excel(r'D:\Dev\0617_P1\EX04.xlsx', usecols=[20,21,22,23,24,25,26,27,28], sheet_name='sam')
# df_01 = df

print(df.head(10))

df_01 = df.iloc[20:]
# df_lft = df.drop(['ex'], axis=1)
print(df_01)
sys.exit()
df_lft = df_lft.drop(['Time'], axis=1)
# print(df_lft.columns.tolist())
print(df_lft.columns)
# sys.exit()
x = df_lft.to_numpy()
y = df_01.to_numpy()
# x = x[0:3]
# y = y[0:3]

# print("x = \n" , x)
# print("y = \n" , y)
# print("x = \n" , x, '\n', x.shape)
# print("y = \n" , y, '\n', y.shape)
# print("test_x = \n" , x_test)
# print("\n\ntrain_x = \n" , x_train)
# print("\n\n\ntest_y = \n" , y_test)
# print("\n\ntrain_y = \n" ,y_train)
# sys.exit()

y_nw = np.eye(15)[y]
y_nw = np.reshape(y_nw, (-1, 15))
print("2Dy = \n" , y_nw, '\n', y_nw.shape)

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
# x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size= 0.2)
# , random_state=1234

# 모델 정의
dnn = keras.Sequential()
dnn.add(Dense(units=128, activation='relu', input_shape=(13,)))
dnn.add(Dense(units=32, activation='relu'))
dnn.add(Dense(units=16, activation='relu'))
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


