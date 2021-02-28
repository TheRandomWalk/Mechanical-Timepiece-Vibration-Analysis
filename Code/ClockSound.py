import numpy
import scipy.signal as signal
import scipy.io.wavfile as wavfile


# Settings

input  = 'Data/Sample.npy'
output = 'Data/Filtered.wav'

inputSamplingRate  = 200_000
outputSamplingRate = 44_100
playSpeed          = 1.
duration           = 60.

filter = True
order  = 5
cutoff = 1_000.


# Code

data = numpy.load(input)
data -= numpy.median(data)

if filter:
    b, a = signal.bessel(order, cutoff / (inputSamplingRate / 2.), 'highpass')
    data = signal.filtfilt(b, a, data)

data = signal.resample(data, int(data.size * outputSamplingRate / (playSpeed * inputSamplingRate)))

data = data[: int(duration * outputSamplingRate)]
data /= numpy.abs(data).max()

data *= ((2. ** 15.) - 1.)
data = data.astype(numpy.int16)

wavfile.write(output, outputSamplingRate, data)
