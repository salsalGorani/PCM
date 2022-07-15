import pandas as pd
import numpy as np

# df = pd.read_excel(r'D:\Dev\0617_P1\EX02.xlsx')
# print(df)

list_tmp = [1,2,3,4,5]
print(list_tmp)

print(list_tmp[:3])

print()
list_tmp2 = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]

print('list_tmp2')
print(np.shape(list_tmp2))
for i in list_tmp2:
    print(i)

list_tmp3 = list_tmp2[:2]
print('list_tmp3')
print(np.shape(list_tmp3))
for i in list_tmp3:
    print(i)

