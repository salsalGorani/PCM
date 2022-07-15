import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import sys

# Prepare data
data = pd.read_excel(r'D:\\Dev\\0617_P1\\EX02.xlsx')
# df = pd.DataFrame(data, columns= ['ex'])
print(data.head(10))


from tensorflow import keras
# from tensorflow.keras.layers import Dense
from keras.layers import Dense



# x값과 일차함수 관계를 갖는 y값 데이터를 만듭니다. (일차함수 y = 2x - 1 사용)
x = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [-21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17]
# print("x = " + type(x), "y = " + type(y))

# sys.exit()


# 1차원 형태의 x 배열을 2차원 형태로 변환합니다.
x_new = np.reshape(x, (-1, 1))
y_new = np.reshape(y, (-1, 1))



# 모델 정의
dnn = keras.Sequential()
dnn.add(Dense(units=1, input_shape=(1,)))
dnn.compile(optimizer='sgd', loss='mse')
# dnn.compile(optimizer='adam', loss='softmax')



# 모델 확인
dnn.summary()


# 모델 학습시키기
dnn.fit(x_new, y_new, epochs=10)



# 새로운 x 데이터에 대한 y값 예측하기
x_test = [[11], [12], [13], [14], [15]]
y_hat = [21., 23., 25., 27., 29.]



# 예측
y_hat_dnn = dnn.predict(x_test)



# 결과 출력
print(y_hat_dnn)
score = dnn.evaluate(x_test, y_hat, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# 벡터값 구현 (더해져야 할 벡터값은 어떻게 구현할 것인지?)