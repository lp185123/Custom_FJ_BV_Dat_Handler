#https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce

import multiprocessing
import time

def worker(x):
    time.sleep(1)
    return x

pool = multiprocessing.Pool()
print(pool.map(worker, range(10)))