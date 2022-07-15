# 아래 그래픽설정을 위함
# 컴퓨터 사정 상 어쩔수없는 코드 "0":내장그래픽, "1":외장그래픽(GTX2060)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import Dense


# Prepare data
df = pd.read_excel(r'D:\Dev\0617_P1\EX02.xlsx')
# print(df)

# df_01 = df[['ex']]                  # 아래(ln.14)의 iloc과 같은 코드
df_01 = df.iloc[:,[12]]               # .iloc[:(=>칼럼전체), [칼럼번호('0'부터~), 칼럼번호 .. ]] => 칼럼추출
# print(df_01)


df_lft = df.drop(['ex'], axis=1)      # .drop => 행열제거(['*제거 행/열', '*'], axis=[**] **=> [0]:행, [1]:열)
df_lft = df_lft.drop(['Time'], axis=1)
x = df_lft.to_numpy()                 # x값 numpy 행렬화 .to_numpy()
y = df_01.to_numpy()                  # y값 numpy 행렬화 .to_numpy()

print("x = \n" , x)
print("y = \n" , y)
# sys.exit()

# print(y)
# y1 = len(y)
print(x.shape)

# # 1차원 형태의 x 배열을 2차원 형태로 변환합니다.
print("x = \n" , x, '\n', x.shape)
print("y = \n" , y, '\n', y.shape)
# x_nw = np.reshape(x, (-1, ))

y_nw = np.eye(15)[y]
# np.eye()는 index값을 기준으로 값을 처리함, 위 코드에서 모체인 'y'는 값을 14까지 가지기 때문에 np.eye(15)로 +1하여 표현해야 했음
# print(y_nw)
y_nw = np.reshape(y_nw, (-1, 15))

# print("2Dx = \n" , x_nw, '\n', x_nw.shape)
print("2Dy = \n" , y_nw, '\n', y_nw.shape)          # shape 확인



# 모델 정의
dnn = keras.Sequential()
dnn.add(Dense(units=64, activation='relu', input_shape=(12,)))
dnn.add(Dense(units=32, activation='relu'))
dnn.add(Dense(units=16, activation='relu'))
dnn.add(Dense(units=15, activation='softmax'))
# dnn.compile(optimizer='sgd', loss='mse')
# dnn.compile(optimizer='adam', loss='softmax')



# # 모델 확인
dnn.summary()

# 3. 실행
# Binary classification, Last-layer activation : sigmoid, Example : Dog vs cat, Sentiemnt analysis(pos/neg)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 알아보자.

# # 모델 학습시키기
dnn.fit(x, y_nw, epochs=100)



# 예측
# y_nw_dnn = dnn.predict(x)



# 결과 출력
# print(y_hat_dnn)
score = dnn.evaluate(x, y_nw, verbose=1)
# print(score)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


