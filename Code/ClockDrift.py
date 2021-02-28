import numpy
import matplotlib.pyplot as pyplot


# Settings

input =  'Data/Ticks.npy'
output = 'Data/Drift.png'

tickRate = 5.
cycle    = 2


# Code

ticks = numpy.load(input)

clockCycle = int(12. * 3600. * tickRate)

accumulatedError = 0.

legend = []

for i in range(cycle):
    error = ticks[: clockCycle]
    error = (error - error[0]) - numpy.arange(error.size) / tickRate

    t = numpy.arange(error.size) / tickRate / 3600.

    pyplot.plot(t, error)

    legend.append('Cycle {:d}'.format(i + 1))

    accumulatedError += error[-1]
    print('Error: Cycle {:d}: {:.3f}, Accumulated: {:.3f} / '.format(i, error[-1], accumulatedError))

    ticks = ticks[clockCycle :]

    if len(ticks) == 0:
        break

pyplot.xlim(0, 12.)
pyplot.xticks([0, 3, 6, 9, 12])

pyplot.xlabel('Time (h)')
pyplot.ylabel('Drift (s)')

pyplot.legend(legend, frameon = False)

pyplot.tight_layout()

pyplot.gcf().set_size_inches(16, 8)
pyplot.savefig(output, dpi = 300)

