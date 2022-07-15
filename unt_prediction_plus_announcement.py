import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense

# Prepare data
list_arr = [[1,1,1,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,0,0,1,1,1]]
x = list_arr
print( "value_of 'list_x' =", x, '\n' )
x = np.reshape(x, (-1, 9))
print( "x_value = < down below >\n" , x, '\n'
       , "Check_shape_of 'x_value' :" , x.shape, '\n' )

y = [0, 1, 2]
print( "value_of_'list_y' =", y )
y_nw = np.eye(3)[y]
print( "One-hot_transformed 'y_value' :\n", y_nw, '\n'
       , "Check_shape_of 'transformed_y_value' :" , y_nw.shape, '\n' )
# sys.exit()

# 모델 정의
print("\nMODEL_DEFINING . .")
dnn = keras.Sequential()
dnn.add(Dense(units=32, activation='relu', input_shape=(9,)))
dnn.add(Dense(units=3, activation='softmax'))
print("COMPLETED.\n")
# # 모델 확인
dnn.summary()

# 3. 실행
print("\nMODEL_COMPILING . .")
dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("COMPLETED.\n")

# 모델 학습시키기
print("START_LEARNING")
dnn.fit(x, y_nw, epochs=100)
print("COMPLETED.\n")

# 예측 데이터 준비
print("PREPARING_PREDICTING_DATAS . .")
x_test = [[0,0,0,1,1,1,1,1,1], [1,1,1,0,0,0,1,1,1], [1,1,1,1,1,1,0,0,0]]
y_hat = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
print("COMPLETED.\n")

# 예측
print("START_PREDICTING")
y_hat = dnn.predict(x_test)
print("\nPredicted results : < Down Below >\n", y_hat)

# 결과 출력
score = dnn.evaluate(x, y_nw, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
