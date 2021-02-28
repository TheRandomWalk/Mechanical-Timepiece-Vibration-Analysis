import time
import numpy
import copy
import threading
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.stream_readers as stream_readers


class DAQ:
    def __init__(self, samplingRate, bufferSize):
        self.task_ = nidaqmx.Task()
        self.task_.ai_channels.add_ai_voltage_chan("MyDAQ1/ai0")
        self.task_.timing.cfg_samp_clk_timing(rate = samplingRate, sample_mode = constants.AcquisitionType.CONTINUOUS, samps_per_chan = bufferSize)

        self.analogMultiChannelReader_ = stream_readers.AnalogMultiChannelReader(self.task_.in_stream)
        self.task_.register_every_n_samples_acquired_into_buffer_event(bufferSize, self.callback)

        self.mutex_ = threading.Lock()


    def __del__(self): 
        self.task_.close()


    def start(self, delay = 0.5):
        self.task_.start()
        time.sleep(delay)
        self.data_ = []


    def stop(self):
        self.task_.stop()


    def download(self):
        self.mutex_.acquire()
        data = copy.deepcopy(self.data_)
        self.data_ = []
        self.mutex_.release()
        return data


    def callback(self, taskHandle, eventType, samples, callbackData):
        buffer = numpy.zeros((1, samples))
        self.analogMultiChannelReader_.read_many_sample(buffer, samples, timeout = constants.WAIT_INFINITELY)
        self.mutex_.acquire()
        self.data_.extend(buffer.tolist()[0])
        self.mutex_.release()
        return 0
