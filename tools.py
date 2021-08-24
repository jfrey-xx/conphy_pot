import numpy as np
import sigproc, copy
from scipy.interpolate import interp1d
import pandas as pd
import warnings

# handling xdf formatted data -- NB: imported from tobecou, in some places changed from using srate to real_srate

# TODO: handle time sync at some point

# TODO: consider shape as a stream (with a dedicated filter)?

def get_stream(xdf_data, stream_name, markers_name = "", stream_type = None, markers_type = None):
   """ Returning *first* selected stream + marker
   xdf_data: structure returned by load_xdf
   stream_name: stream to fetch from data
   markers_name: if set, will as well return marker
   stream_type, markers_type: if set, will also filter by stream type
   
   Returns a dict with keys:
   data: nump array containing time series
   timestamps: time of each data (as from system clock?)
   nbchans: number of channels in data
   srate: declared sampling rate of the stream
   real_srate: real sampling rate of the whole stream
   xdf_real_srate: real sampling rate of the whole stream, as computed by pyxdf (can handle better change of clock)
   markers_name: optionnal list of markers
   nbmarkers: how many of them
   markers_idx: corresponding index of the markers in data (will be adjusted closest to timestamp)
   markers_shift: diff between original markers' timestamps and current data
   markers_timestamps: original timestamps for markers
   
   NB: optionnal fields that could be added by set_shape:
       shape: shape of a breathing pattern
       codes: associated code for high/low peaks (see features.py)
   
   WARNING: suppose that streams timestamps are synced. Assume constant rate...
   """
   
   data_stream = None
   markers_stream = None
   
   # get the stream of interest
   for stream in xdf_data[0]:
       if stream['info']['name'][0] == stream_name:
           if stream_type is None or stream_type == stream['info']['type'][0]:
               data_stream = stream
       elif markers_name != "" and stream['info']['name'][0] == markers_name:
           if markers_type is None or markers_type == stream['info']['type'][0]:
               markers_stream = stream

   # at least we should get the stream by now
   if data_stream == None:
       return None
       
   res = {}
   res['data'] = data_stream['time_series']
   res['timestamps'] = data_stream['time_stamps']
   res['nbchans']  = int(data_stream['info']['channel_count'][0])
   res['srate'] = float(data_stream['info']['nominal_srate'][0])
   time_range = res['timestamps'][-1] - res['timestamps'][0]
   nb_data = res['data'].shape[0]
   res['real_srate'] = float(nb_data) / time_range
   res['xdf_real_srate'] = data_stream['info']['effective_srate']
   # holder or markers
   res['markers'] = np.array([])
   res['nbmarkers'] = 0
   res['markers_idx'] = np.array([])
   res['markers_shift'] = np.array([])
   
   # markers will be left empty if we don't find anything
   if markers_stream == None:
       return res
   
   res['markers'] = np.concatenate(markers_stream['time_series']) # each elemet are in a separate list, streamline that
   res['nbmarkers'] = len(markers_stream['time_series'])
   res['markers_timestamps'] = markers_stream['time_stamps']
   [res['markers_idx'], res['markers_shift']] = fit_markers(res['timestamps'], res['markers_timestamps'])
   
   return res

def fit_markers(timestamps, markers_timestamps):
    """ Find indexes matching markers' timestamps within data.
    
    timestamps: timestamps of the original data
    markers_timestamps: timestamps of the markers to fit
    
    returns [makers_idx, markers_shift]: index of markers within timestamps, diff between marker timestamps and original one
    """
    nb_markers = len(markers_timestamps)
    markers_idx = np.zeros(nb_markers)
    markers_shift = np.zeros(nb_markers)
    # will look for nearest data index which timestamps match markers'
    for i in range(nb_markers):
        stamp = markers_timestamps[i]
        idx = np.abs(timestamps-stamp).argmin() # trick from https://stackoverflow.com/a/2566508
        markers_idx[i] = idx
        markers_shift[i] = timestamps[idx] - stamp
    return [markers_idx, markers_shift]
   
def bandpass_stream(stream, lowcut, highcut, order = 5):
    """ Applying band-pass filtering to all the channels of a stream as returned by get_stream()
    lowcut, highcut: frequency interval
    order: order of the butterworth filter
    WARNING: the shape or code, if any, will be untouched, should be computed again if necessary
    """
    filtered = copy.deepcopy(stream)
    for i in range(filtered['nbchans']):
        filtered['data'][:,i] = sigproc.butter_bandpass_filter(stream['data'][:,i], lowcut, highcut, int(stream['real_srate']), order=order)
    return filtered

def bandpass_array(array, srate, lowcut, highcut, order = 5):
    """ Applying band-pass filtering to raw channels of numpy 2D array (by second column)
    srate: sampling rate of data
    lowcut, highcut: frequency interval
    order: order of the butterworth filter
    WARNING: not tested
    """
    # only 2D array
    assert(len(array.shape) == 2)
    filtered = copy.deepcopy(array)
    for i in range(filtered.shape[1]):
        filtered[:,i] = sigproc.butter_bandpass_filter(array[:,i], lowcut, highcut, srate, order=order)
    return filtered
    
def shift_stream(stream, value):
    """ Will add value to all stream data. Usefull for fake breath, to shift 0..1 interval
    stream: dict as returned by get_stream()
    value: value which will be added to all channels
    WARNING: the shape or code, if any, will be untouched, should be computed again if necessary
    """
    filtered = copy.deepcopy(stream)
    filtered['data'] += value
    return filtered
    
def upsample_stream(stream, factor, method = "linear"):
    """ Increasing sampling rate of data stream by factor.
    Will interpolate data (and timestamps).
    As a side effect: may correct for data skewed during acquisition (will create linear timestamps).
   
    stream: dict as returned by get_stream()
    factor: int number multiply sampling rate with (e.g. 5 to convert from 24 to 120)
    method: method to be used by scipy interp1d for interpolling, "linear" (default) or "cubic" (NB: cubic is memory hungry)
    
    return: a stream
    
    WARNING: should take place before band-pass filtering
    TODO: might look at resampling method, e.g. scipy
    FIXME: loose info about shape/codes if any
    """
    # work on a copy of the stream
    res = copy.deepcopy(stream)
    # discard shape/code, we don't know how to handle that yet
    if 'shape' in stream:
        warnings.warn('shape not surppored for upsample, discard')
        stream.pop('shape', None)
    if 'codes' in stream:
        warnings.warn('codes not surppored for upsample, discard')
        stream.pop('codes', None)

    # sanitizing input
    factor = int(factor)
    
    # init new data holder
    nb_data_orig = stream['data'].shape[0]
    nb_data = nb_data_orig * factor
    res['data'] = np.zeros((nb_data, res['nbchans']))
    
    # upscale timestamps
    res['timestamps'] = np.linspace(stream['timestamps'][0], stream['timestamps'][-1] , nb_data)
    
    # upscale for each channel
    for c in range(res['nbchans']):
        # get corresponding channel, compute interpolation, apply, save
        cf = interp1d(stream['timestamps'], stream['data'][:, c], kind=method)
        res['data'][:, c] = cf(res['timestamps'])
        
    # update info
    time_range = stream['timestamps'][-1] - stream['timestamps'][0]
    res['real_srate'] = float(nb_data) / time_range
    res['srate'] = stream['srate'] * factor
    
    # re-sync markers, if any
    if res['nbmarkers'] > 0:
        [res['markers_idx'], res['markers_shift']] = fit_markers(res['timestamps'], res['markers_timestamps'])
   
    return res
    
def epoching(stream, labels, length):
    """ Epoching a stream, slicing data from labels start 
    
    stream: dict as returned by get_stream()
    labels: list of markers of interest
    length: epochs length (in seconds). WARNING: will use REAL sampling rate, not nominal sampling rate, for better match.
    
    return: an array of epochs, sorted in time, each one being
    epoch['data']: array of data bins (values * chans). Shorten an epoch that would go beyond stream data size.
    epoch['marker']: corresponding label
    epoch['srate']: srate of chunk (will be same as stream, facilitate handling later on)
    epoch['timestamp']: when the epoch started in the original data
    if set in original stream: epoch['shape'] and epoch['codes'], as extracted by features.extract_breath_shape()
    TODO: use same structure as in stream
    """  
    # won't get data if no markers
    if stream['nbmarkers'] == 0:
        return np.array([[]])
    
    # we will build a list of epochs
    epochs = []
    # for each labels add epoch to stack
    for l in labels:
        indexes = stream['markers_idx'][stream['markers']  == l]
        # retrieve corresponding slices
        for idx in indexes:
            start = int(idx)
            epoch_size = int(length * stream['real_srate'])
            stop = start+epoch_size
            # prevent overbound
            if idx+epoch_size > len(stream['data']):
                stop = len(stream['data'])
            epoch = {}
            epoch['data'] = stream['data'][start:stop,:]
            epoch['marker'] = l
            epoch['srate'] = stream['srate']
            epoch['real_srate'] = stream['real_srate']
            epoch['timestamp'] = stream['timestamps'][start]
            # optionnal fields
            if 'shape' in stream:
                epoch['shape'] = stream['shape'][start:stop]
            if 'codes' in stream:
                epoch['codes'] = stream['codes'][start:stop]

            epochs.append(epoch)

    # sort by timestamp
    epochs = sorted(epochs, key=lambda k: k['timestamp']) 
    # everything's numpy!
    return np.array(epochs)

def merge_epochs_features(epochs, feats):
    """ Create a Pandas data frame merging epochs and features
    epochs: a list of epochs as returned by epoching()
    feats: a dict of features as returned by features.extract_breath_features()    
    returns: data frame concatenating data/label/timestamp/features, with an adittionnal "num" field for epoch number
    
    NB: epochs and feats lists must have the same size
    TODO: use shape and codes if included?
    FIXME: after repo split, might be now useless code,
    """
    export = pd.DataFrame()
    if len(epochs) != len(feats):
        return export
    # append to data frame each epoch, data, label of epoch, index, and feattures
    for i in range(len(epochs)):
        e = epochs[i]  
        edf = pd.DataFrame(e['data'])
        edf['label'] = e['marker']
        edf['num'] = i
        edf['timestamp'] = e['timestamp']
        # fetch corresponding features
        edf['amplitude'] = feats[i]['amplitude']
        edf['breath_in'] = feats[i]['breath_in']
        edf['breath_out'] = feats[i]['breath_out']
        edf['breaths'] = feats[i]['breaths']
        edf['hold_in'] = feats[i]['hold_in']
        edf['hold_out'] = feats[i]['hold_out']
        export = pd.concat([export, edf])
    return export

def set_shape(stream, shape, codes):
    """ Associate a breathing shape extrated from features to a stream.
    stream: struct as returned by get_stream
    shape, codes: as returned by feature.sextract_breath_shape()
    returns: a duplicated stream updated with new fieds
    """
    # work on a copy of the stream
    res = copy.deepcopy(stream)
    # check format
    if shape.shape[0] != res['data'].shape[0]:
        raise NameError('Wrong length for shape: ' + str(shape.shape[0]) + " instead of " + str(res['data'].shape[0]))
    elif codes.shape[0] != res['data'].shape[0]:
        raise NameError('Wrong length for codes: ' + str(codes.shape[0]) + " instead of " + str(res['data'].shape[0]))
    res['shape'] = shape
    res['codes'] = codes
    return res
    
def get_epoch(stream, marker_start, marker_stop):
    """
    Will return a stream with data selected between marker_stant and marker_end.
    Copy data associated to stream except for markers.
    NB: real srate recopmuted for this particular epoch
    NB: only take into actount first appearance of each makers.
    """
    start = stream['markers_idx'][stream['markers']  == marker_start]
    stop = stream['markers_idx'][stream['markers']  == marker_stop]
    
    # won't get data if no markers
    if len(start) == 0 or len(stop) == 0:
        return np.array([[]])
    
    start_idx = int(start[0])
    stop_idx = int(stop[0])
    
    epoch = {}
    epoch['data'] = stream['data'][start_idx:stop_idx,:]
    epoch['srate'] = stream['srate']
    epoch['nbchans'] = stream['nbchans']
    epoch['timestamps'] = stream['timestamps'][start_idx:stop_idx]
    time_range = epoch['timestamps'][-1] - epoch['timestamps'][0]
    nb_data = epoch['data'].shape[0]
    epoch['real_srate'] = float(nb_data) / time_range
    
    return epoch
    