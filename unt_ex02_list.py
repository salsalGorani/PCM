import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense

# Prepare data
# list_arr = [[1,1,1],[0,0,0],[0,0,0], [0,0,0],[1,1,1],[0,0,0], [0,0,0],[0,0,0],[1,1,1]]
list_arr = [[1,1,1,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,0,0,1,1,1]]
x = list_arr
print(x)
x = np.reshape(x, (-1, 9))
print("x_value = \n" , x, '\n', x.shape)

y = [0, 1, 2]
y_nw = np.eye(3)[y]
print(y_nw, '\n', y_nw.shape)
# sys.exit()

# 모델 정의
dnn = keras.Sequential()
dnn.add(Dense(units=32, activation='relu', input_shape=(9,)))
dnn.add(Dense(units=3, activation='softmax'))

# # 모델 확인
dnn.summary()

# 3. 실행
dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 학습시키기
dnn.fit(x, y_nw, epochs=100)

# 예측 데이터 준비
x_test = [[0,0,0,1,1,1,1,1,1], [1,1,1,0,0,0,1,1,1], [1,1,1,1,1,1,0,0,0]]
y_hat = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

# 예측
y_hat = dnn.predict(x_test)
print(y_hat)

# 결과 출력
score = dnn.evaluate(x, y_nw, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
