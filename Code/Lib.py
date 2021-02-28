import copy
import numpy
import numpy.fft as fft
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate


def findTicks(m, blockSize, scanSize):
    n = numpy.array(m)
    n = numpy.abs(n - n.mean())

    halfScanSize = scanSize >> 1

    peak = numpy.argmax(n[halfScanSize : halfScanSize + blockSize]) + halfScanSize

    index = []

    while True:
        start = peak - halfScanSize
        stop  = peak + halfScanSize

        if stop >= n.size:
            break

        peak = numpy.argmax(n[start: stop]) + start
        index.append(peak)
            
        peak = peak + blockSize

    return numpy.array(index)

