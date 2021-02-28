import numpy
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as pyplot

import Lib


# Settings

input                       = 'Data/Sample.npy'
outputWaveformUnfiltered    = 'Data/WaveformUnfiltered.png'
outputWaveformFiltered      = 'Data/WaveformFiltered.png'
outputSpectrogramUnfiltered = 'Data/SpectrogramUnfiltered.png'
outputSpectrogramFiltered   = 'Data/SpectrogramFiltered.png'

samplingRate = 200_000
tickRate     = 5.
sensitivity  = 0.02

tickBlockSize = int(samplingRate / tickRate)
tickScanSize  = int(0.1 * tickBlockSize)

order  = 5
cutoff = 1_000.

tick = 10

waveformMin = -25.
waveformMax =  25.

vminPercentile = .1
vmaxPercentile = 99.9


# Code: Preprocess

print('Preprocessing...')

unfiltered = numpy.load(input)
unfiltered -= numpy.median(unfiltered)
unfiltered = unfiltered / sensitivity

t = numpy.arange(unfiltered.size) / samplingRate

location = Lib.findTicks(unfiltered, tickBlockSize, tickScanSize)

b, a = signal.bessel(order, cutoff / (samplingRate / 2), 'highpass')
filtered = signal.filtfilt(b, a, unfiltered)

start = location[location.size >> 1] - (tickBlockSize >> 1)
stop  = location[(location.size >> 1) + tick] + (tickBlockSize >> 1)


# Code: Unfiltered waveform

print('Generating unfiltered waveform...')

pyplot.plot(t[start : stop] - t[start], unfiltered[start : stop])

pyplot.ylim(waveformMin, waveformMax)

pyplot.xlabel('Time (s)')
pyplot.ylabel('Acceleration ($m/s^2$)')

pyplot.gcf().set_size_inches(16, 4)
pyplot.savefig(outputWaveformUnfiltered, dpi = 300)
pyplot.clf()


# Code: Filtered waveform

print('Generating filtered waveform...')

pyplot.plot(t[start : stop] - t[start], filtered[start : stop])

pyplot.ylim(waveformMin, waveformMax)

pyplot.xlabel('Time (s)')
pyplot.ylabel('Acceleration ($m/s^2$)')

pyplot.gcf().set_size_inches(16, 4)
pyplot.savefig(outputWaveformFiltered, dpi = 300)
pyplot.clf()


# Code: Unfiltered spectrogram

print('Generating unfiltered spectrogram...')

frequency, time, spectrogram = signal.spectrogram(unfiltered[start : stop], samplingRate, nperseg = 1024)
spectrogram = numpy.log10(spectrogram)
frequency /= 1_000

vmin = numpy.percentile(spectrogram, vminPercentile)
vmax = numpy.percentile(spectrogram, vmaxPercentile)

pyplot.pcolormesh(time, frequency, spectrogram, shading = 'nearest', vmin = vmin, vmax = vmax, cmap = 'inferno')

pyplot.xlabel('Time (s)')
pyplot.ylabel('Frequency (kHz)')

pyplot.gcf().set_size_inches(16, 8)
pyplot.savefig(outputSpectrogramUnfiltered, dpi = 300)
pyplot.clf()


# Code: filtered spectrogram

print('Generating filtered spectrogram...')

frequency, time, spectrogram = signal.spectrogram(filtered[start : stop], samplingRate, nperseg = 1024)
spectrogram = numpy.log10(spectrogram)
frequency /= 1_000

pyplot.pcolormesh(time, frequency, spectrogram, shading = 'nearest', vmin = vmin, vmax = vmax, cmap = 'inferno')

pyplot.xlabel('Time (s)')
pyplot.ylabel('Frequency (kHz)')

pyplot.gcf().set_size_inches(16, 8)
pyplot.savefig(outputSpectrogramFiltered, dpi = 300)
pyplot.clf()
