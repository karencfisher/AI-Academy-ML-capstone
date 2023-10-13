import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

from LSTM.Config import Options as opts


def process_data(pos_data, neg_data, max_steps=500):
    pos_data, neg_data = split_data(pos_data, neg_data)
    pos_data = filter_outlier_sequences(pos_data, max_steps)
    neg_data = filter_outlier_sequences(neg_data, max_steps)
    raw_data = pd.concat([pos_data, neg_data])

    pos_ids = pos_data['VisitIdentifier'].unique().tolist()
    neg_ids = neg_data['VisitIdentifier'].unique().tolist()
    ids = pos_ids + neg_ids

    data = [raw_data[raw_data['VisitIdentifier'] == i]
                 [opts.numerical_feat].values.tolist() for i in ids]
    data = pad_sequences(data, padding='post', 
                         value=-10, maxlen=max_steps).tolist()

    labels = [1 for _ in range(len(pos_ids))]
    labels.extend(0 for _ in range(len(neg_ids)))
    return data, labels


def filter_outlier_sequences(input_data, threshold):
    counts = input_data.groupby(by='VisitIdentifier')['MinutesFromArrival'].count()
    counts.sort_values(ascending=False)
    outliers = counts[counts > threshold].index.values.tolist()
    output_data = input_data[~input_data['VisitIdentifier'].isin(outliers)]
    return output_data
    

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
  



    


