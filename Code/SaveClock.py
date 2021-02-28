import numpy
import time
import scipy.signal as signal
import matplotlib.pyplot as pyplot

import DAQ


# Settings

filename     = 'Data/ClockRaw_{:04d}.npy'
samplingRate = 200_000
samples      = 60 * samplingRate


# Code

daq = DAQ.DAQ(samplingRate, 2_000_000)
daq.start()

data = []
iterator = 0

t0 = time.time()

while True:
    time.sleep(.01)

    data.extend(daq.download())

    if len(data) >= samples:
        print('[{:.2f} h] Saving data to \'{:s}\'...'.format((time.time() - t0) / 3_600, filename.format(iterator)))

        numpy.save(filename.format(iterator), data[: samples])
        
        data = data[samples :]
        iterator += 1
