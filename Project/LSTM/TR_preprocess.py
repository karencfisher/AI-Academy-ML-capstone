import numpy as np
import math
from tqdm import tqdm

from LSTM.Config import Options as opts
from LSTM.TemporalAbstraction import discretization, make_MSS


def preprocess(pos_data, neg_data, filter=True, encode=True, max_times=40000):
    pos_data, neg_data = split_data(pos_data, neg_data)

    pos_data = filter_outlier_sequences(pos_data, max_times)
    neg_data = filter_outlier_sequences(neg_data, max_times)

    # First, we will use the RTP code to get Multivariate State Sequences
    if encode:
        for value in opts.numerical_feat:
            temperature = value == 'Temperature'
            pos_data[value], neg_data[value] = discretization(pos_data[value], 
                                                            neg_data[value],
                                                            temperature=temperature) 
    MSS_pos = make_MSS(pos_data)
    MSS_neg = make_MSS(neg_data)
    MSS = MSS_pos + MSS_neg

    # Then, we will use them to reconstruct a consistent time line for
    # observed values, e.g., every 10 minutes.
    dataset = reconstruct_timeline(MSS, max_times, opts.interval)
    labels = [1 for _ in range(len(MSS_pos))]
    labels += [0 for _ in range(len(MSS_neg))]
    return dataset, np.array(labels)

def filter_outlier_sequences(input_data, threshold):
    counts = input_data.groupby(by='VisitIdentifier')['MinutesFromArrival'].max()
    counts.sort_values(ascending=False)
    outliers = counts[counts > threshold].index.values.tolist()
    output_data = input_data[~input_data['VisitIdentifier'].isin(outliers)]
    return output_data

def reconstruct_timeline(MSS, max_times, interval):
    feature_map = {feat: indx for indx, feat in enumerate(opts.numerical_feat)}
    n_visits = len(MSS)
    n_features = len(opts.numerical_feat)
    max_times = int(math.ceil(max_times / opts.interval))
    data = np.full((n_visits, max_times, n_features), -10, dtype=int)
    for visit_indx, MSS_events in enumerate(MSS):
        for event in MSS_events:
            feature, value, start, stop = event
            feature_indx = feature_map[feature]
            start = math.ceil(start / interval)
            stop = math.ceil(stop / interval)
            data[visit_indx, start:stop, feature_indx] = int(value)
    return data

def split_data(pos_events, neg_events):
    if opts.early_prediction > 0 and opts.alignment == 'right':
        pos_cut = pos_events[pos_events.EventTime - pos_events[opts.timestamp_variable] >= 
                                                    opts.early_prediction * 60]
        neg_cut = neg_events[neg_events.LastMinute - neg_events[opts.timestamp_variable] >= 
                                                    opts.early_prediction * 60]
        if opts.observation_window:
            pos_cut = pos_cut[pos_cut.EventTime - pos_cut[opts.timestamp_variable] <= 60 * 
                                        (opts.observation_window + opts.early_prediction)]
            neg_cut = neg_cut[neg_cut.LastMinute - neg_cut[opts.timestamp_variable] <= 60 * 
                                        (opts.observation_window + opts.early_prediction)]

    elif opts.observation_window and opts.alignment == 'left':
        pos_cut = pos_events[pos_events[opts.timestamp_variable] <= 
                                            opts.early_prediction * 60]
        neg_cut = neg_events[neg_events[opts.timestamp_variable] <= 
                                            opts.early_prediction * 60]
        
    if opts.settings == 'trunc':
        pos_events = pos_cut
        neg_events = neg_cut

    return pos_events, neg_events
  



    


