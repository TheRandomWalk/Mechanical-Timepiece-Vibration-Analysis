import numpy
import tqdm

import Lib


#Settings

input  = 'Data/ClockRaw_{:04d}.npy'
output = 'Data/Ticks.npy'

file         = 24 * 60
samplingRate = 200_000.
tickRate     = 5.

tickBlockSize = int(samplingRate / tickRate)
tickScanSize  = int(0.1 * tickBlockSize)


# Code

vibration = []
vibrationOffset = 0
ticks = []

clock = []
clockOffset = 0
zeros = []

for i in tqdm.tqdm(range(file), desc = 'Processing data', ascii = True, bar_format = '{desc}: |{bar}| [{elapsed}<{remaining}]'):
    m = numpy.load(input.format(i))

    vibration.extend(m.tolist())
    newTicks = Lib.findTicks(vibration, tickBlockSize, tickScanSize)
    ticks.extend((vibrationOffset + newTicks).tolist())
    vibrationRemove = min(newTicks[-1] + (tickBlockSize >> 1), len(vibration))
    vibration = vibration[vibrationRemove:]
    vibrationOffset += vibrationRemove

ticks = numpy.array(ticks) / samplingRate
diff = numpy.diff(ticks)

print()
print('TICKS')
print('Total: {:,d}'.format(len(ticks)))
print('Mean:  {:.3f} ms'.format(diff.mean() * 1_000.))
print('Min:   {:.3f} ms'.format(diff.min()  * 1_000.))
print('Max:   {:.3f} ms'.format(diff.max()  * 1_000.))

print()
print('Saving ticks...')
numpy.save(output, ticks)
