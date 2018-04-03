import sys
f = open('tmp.txt', 'ab')
while 1:
    raw_input()
    f.write('this is a test\n')
    f.flush()
f.close()
