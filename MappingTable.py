list_map = [-1, 1, 0, 2, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 1]

with open('unt_mt01.txt', 'r') as fptr:
    for i in fptr.readlines():
        i = int(i.strip())
        print(i, list_map[i])
