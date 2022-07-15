import numpy as np
# list_0 = ['0']*10
# list_str = ', '.join(list_0)
#
# print(list_0)
# print(list_str)

# 0 으로 구성된 리스트 만드  는 방법
list_0 = [0]*10
print(list_0)

list_00 = np.zeros(10)
print(list_00)

# 1 ~ 10 의 값으로 구성된 리스트를 만드세요.
list_10 = [i for i in range(1, 11)]
# list_tmp = list(list_10)
# print(len(list_tmp))
print(len(list_10))
print(type(list_10))

# 슬라이싱
print(list_10[:7])
print(list_10[7:])

str_url = 'www.google.com'
# idx = str_url.find('.')
# print(idx)
idx = str_url.index('.')
print(idx)
print(str_url[4:10])

# 회사 이름만 뽑아 볼래요?
idx = str_url.split('.')
print(idx[1])





