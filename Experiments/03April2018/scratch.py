import re
import numpy as np

K = '''
[[-135.56929907   -8.81640038   31.87093922]
[  24.70393703  145.55461666   38.67206181]
[   0.            0.            0.        ]] '''

print ( float('145'))

thetas = ''' 
 [[  0.      -1.4375   0.    ]
 [ -0.0625  -1.3125   0.0625]
 [  0.      -1.6875   0.    ]
 [ -0.0625  -3.1875  -0.0625]
 [ -0.0625  -3.0625  -0.0625]
 [ -0.1875  -2.6875  -0.1875]
 [ -0.1875  -5.      -0.125 ]
 [ -0.125   -4.6875  -0.0625]
 [ -0.1875  -5.125   -0.125 ]
 [ -0.25    -6.5625  -0.25  ]
 [ -0.25    -6.9375  -0.1875]
 [ -0.3125  -6.75    -0.25  ]
 [ -0.3125  -8.5625  -0.25  ]
 [ -0.3125  -8.875   -0.125 ]
 [ -0.25    -8.      -0.1875]
 [ -0.4375 -10.3125  -0.375 ]
 [ -0.4375 -10.8125  -0.375 ]
 [ -0.4375 -10.375   -0.3125]
 [ -0.5    -12.5     -0.5625]
 [ -0.4375 -11.375   -0.5   ]
 [ -0.5625 -11.4375  -0.5   ]] '''
torques = '''
 [[    8.   -92.     0.]
 [   16.  -184.     0.]
 [   24.  -276.     0.]
 [   32.  -368.     0.]
 [   40.  -460.     0.]
 [   48.  -552.     0.]
 [   56.  -644.     0.]
 [   64.  -736.     0.]
 [   72.  -828.     0.]
 [   80.  -920.     0.]
 [   88. -1012.     0.]
 [   96. -1104.     0.]
 [  104. -1196.     0.]
 [  112. -1288.     0.]
 [  120. -1380.     0.]
 [  128. -1472.     0.]
 [  136. -1564.     0.]
 [  144. -1656.     0.]
 [  152. -1748.     0.]
 [  160. -1840.     0.]
 [  168. -1932.     0.]] '''
import collections

def print_to_list(astr):
    # take output of a print(alist), turn into numpy array
    c = collections.Counter(astr)
    numtimes = c['['] -1
    alist = re.split("\s+", astr.replace('[','').replace(']',''))
    alist = list(filter(None, alist))
    alist = [float(z) for z in alist]
    arr = np.array(alist,dtype=float).reshape(numtimes, -1)
    return arr

# def pprint(astr,z):
    # print(astr)
    # print(z)

# a = [1,2,3]
# pprint('a', eval('a'))

K = print_to_list(K)
thetas= print_to_list(thetas)
torques = print_to_list(torques)

#print('thetas\n', thetas)
print('torques\n',torques)
print('K\n', K)
print('torque est')
torq_est = np.dot(K, thetas.T)
#print(torq_est.shape)
#print(torq_est.T)
#print(torq_est.T - torques)
mse = ((torq_est.T - torques) ** 2).mean(axis=0)
print('rmse', np.sqrt(mse))
