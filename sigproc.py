# misc functions for filtering

from scipy.signal import butter, lfilter
import numpy as np

# ease use of filtering
# http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
# https://stackoverflow.com/a/2652425
def signchange(input_array):
    """ return an array with 0 and 1 when the sign changes  """ 
    a = np.array(input_array)
    asign = np.sign(a)
    return ((np.roll(asign, 1) - asign) != 0).astype(int)
