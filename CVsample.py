import time
import numpy as np
import sys



while True:
    sampledata = np.random.randint(-60,60,45)
    print(sampledata)
    print('EOF')
    sys.stdout.flush()
    time.sleep(5)
