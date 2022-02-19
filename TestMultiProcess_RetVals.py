#https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce

import multiprocessing
import time
import random

def wake():
    print("oh no")

def worker(x):
    print("SLeeeping",x.arse)
    
    time.sleep(x.arse)
    wake()
    x.arse="FUCK OFF" + str(x.arse)
    return x
class plop():
    def __init__(self):
        self.arse=random.random()*10

if __name__ == "__main__":
    pool = multiprocessing.Pool()

    listJobs=[]
    for I in range(4):
        plopplop=plop()
        listJobs.append(plopplop)
    results=(pool.map(worker,listJobs))
    plop=1