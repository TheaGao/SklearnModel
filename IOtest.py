b = [0] * 10
i = 0
a = 1
while a != 0:
    a = raw_input('please input what you want: ')
    a = int(a)
    if a == 0:
        break
    b[i] = a
    i = i + 1
print b[0:i]
