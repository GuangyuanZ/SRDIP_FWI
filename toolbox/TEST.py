import numpy as np
from multiprocessing import Pool
import time
t1 = time.time()
a = np.zeros((5,5,2))
pool = Pool(4)
for i in range(4):
    homepath = ('/home/zgy/桌面/T_num/rec_num'+str(i+1)+'.dat')
    a[i,::] = np.loadtxt(homepath)
pool.close()
pool.join()
t2 = time.time()
print(t2-t1)
print(np.shape(a))
