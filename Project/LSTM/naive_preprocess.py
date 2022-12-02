import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

from LSTM.Config import Options as opts


def process_data(pos_train, neg_train, pos_test, neg_test, filter=True):
    pos_train_data, neg_train_data = split_data(pos_train, neg_train)
    if filter:
        pos_train_data, pos_count = filter_outlier_sequences(pos_train_data)
        neg_train_data, neg_count = filter_outlier_sequences(neg_train_data)
    raw_train_data = pd.concat([pos_train_data, neg_train_data])
    train_count = max(pos_count, neg_count)

    pos_test_data, neg_test_data = split_data(pos_test, neg_test)
    if filter:
        pos_test_data, pos_count = filter_outlier_sequences(pos_test_data)
        neg_test_data, neg_count = filter_outlier_sequences(neg_test_data)
    raw_test_data = pd.concat([pos_test_data, neg_test_data])
    test_count = max(pos_count, neg_count)

    pos_train_ids = pos_train_data['VisitIdentifier'].unique().tolist()
    neg_train_ids = neg_train_data['VisitIdentifier'].unique().tolist()
    train_ids = pos_train_ids + neg_train_ids
    print(f'training visits: {len(train_ids)}')

    pos_test_ids = pos_test_data['VisitIdentifier'].unique().tolist()
    neg_test_ids = neg_test_data['VisitIdentifier'].unique().tolist()
    test_ids = pos_test_ids + neg_test_ids
    print(f'test visits: {len(test_ids)}')

    train_data = [raw_train_data[raw_train_data['VisitIdentifier'] == 
            i][opts.numerical_feat].values.tolist() for i in train_ids]
    test_data = [raw_test_data[raw_test_data['VisitIdentifier'] == 
            i][opts.numerical_feat].values.tolist() for i in test_ids]

    max_len = max(train_count, test_count)
    train_data = pad_sequences(train_data, padding='post', 
                               value=-10, maxlen=max_len).tolist()
    test_data = pad_sequences(test_data, padding='post', 
                               value=-10, maxlen=max_len).tolist()

    train_labels = [1 for _ in range(len(pos_train_ids))]
    train_labels.extend(0 for _ in range(len(neg_train_ids)))
    test_labels = [1 for _ in range(len(pos_test_ids))]
    test_labels.extend(0 for _ in range(len(neg_test_ids)))
    return train_data, train_labels, test_data, test_labels


def filter_outlier_sequences(input_data, threshold=900):
    counts = input_data.groupby(by='VisitIdentifier')['MinutesFromArrival'].count()
    counts.sort_values(ascending=False)
    outliers = counts[counts > threshold].index.values.tolist()
    output_data = input_data[~input_data['VisitIdentifier'].isin(outliers)]
    counts = counts[counts <= threshold]
    max_len = counts[counts <= threshold].max()
    return output_data, max_len
    

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
  



    


