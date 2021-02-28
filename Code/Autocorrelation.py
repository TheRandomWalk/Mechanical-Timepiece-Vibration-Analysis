
import numpy
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as pyplot
import tqdm
import multiprocessing

import Lib


# Settings

input  = 'Data/Sample.npy'
output = 'Data/Autocorrelation.png'

samplingRate = 200_000

lowLimit = 1_000
highLimit = 0


windowTime = 6.
slideTime  = 45.

threads = 16
batch   = 500


# Functions

def correlate(baseSlice, spectrogram, windowUnit, start, stop):
    correlation = []

    for i in range(start, stop):
        slice = spectrogram[:, i : i + windowUnit].flat[:]
        correlation.append(numpy.corrcoef(baseSlice, slice)[0, 1])

    return correlation


# Code: Preprocess

if __name__ == '__main__':
    print('Preprocessing...')

    unfiltered = numpy.load(input)
    unfiltered -= numpy.median(unfiltered)

    frequency, time, spectrogram = signal.spectrogram(unfiltered, samplingRate, nperseg = 64, noverlap = 1)

    time -= time.min()

    if lowLimit != 0:
        spectrogram = spectrogram[frequency > lowLimit]
        frequency   = frequency[frequency > lowLimit]

    if highLimit != 0:
        spectrogram = spectrum[frequency < highLimit]
        frequency   = frequency[frequency < highLimit]

    windowUnit = int(numpy.round(windowTime / time[1]))
    slideUnits = int(numpy.round(slideTime / time[1]))

    baseSlice = spectrogram[:, 0 : windowUnit].flat[:]

    correlation = []
    parameter = []

    pool = multiprocessing.Pool(threads)

    for i in range((slideUnits + batch - 1) // batch):
        start = i * batch
        stop = min(start + batch, slideUnits)

        parameter.append([baseSlice, spectrogram, windowUnit, start, stop])

    for i in tqdm.tqdm(range((len(parameter) + threads - 1) // threads), desc = 'Computing', ascii = True):
        start = i * threads
        stop = min(start + threads, len(parameter))

        ret = pool.starmap(correlate, parameter[start : stop])
        
        expanded = [item for sublist in ret for item in sublist]

        correlation.extend([item for sublist in ret for item in sublist])
        
    correlation = numpy.array(correlation)

    pyplot.plot(time[:correlation.size], correlation)
    
    pyplot.xlabel('Displacement (s)')
    pyplot.ylabel('Pearson correlation')

    pyplot.gcf().set_size_inches(16, 8)
    pyplot.savefig(output, dpi = 300)
