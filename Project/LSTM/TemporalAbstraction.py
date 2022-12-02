import numpy as np
import pandas as pd
from tqdm import tqdm

from LSTM.Config import Options as opts


def discretization(features_1, features_0, temperature=False):
    values = pd.concat([features_1, features_0])

    if temperature:
        bins = [0, 36.1, 37.3, values.max() + 1]
        labels = ['2', '3', '4']
    else:
        percentiles = [10, 25, 75, 90, 100]
        labels = ['1', '2', '3', '4', '5']
        bins = [0] + [np.percentile(values, p) for p in percentiles]
        bins[-1] = values.max() + 1

    f = lambda x: labels[i] if not isinstance(x, str) and \
                  bins[i] <= x < bins[i+1] else x

    for i in range(len(bins) - 1):
        features_1 = features_1.apply(f)
        features_0 = features_0.apply(f)
    return features_1, features_0

def make_MSS(events):
    assert len(events) > 0, 'empty sequence'
    MSS = []
    VisitIDs = events['VisitIdentifier'].unique()
    for visit_id in tqdm(VisitIDs, desc='generate'):
        data = events[events['VisitIdentifier'] == visit_id]
        mss = MultiStateSequence(data, opts.numerical_feat, opts.timestamp_variable)
        MSS.append(mss)
    return MSS

def MultiStateSequence(data, features, time_stamp):
    MSS = []
    for feature in features:
        MSS.extend(state_generation(data, feature, time_stamp))
    MSS.sort(key=lambda x: x[2])
    return MSS

def state_generation(discrete_values, feature, time_stamp):
    state_intervals = []
    prev_state = np.nan
    for _, val in discrete_values.iterrows():
        if pd.isnull(prev_state):
            prev_state = val[feature]
            start_state = end_state = val[time_stamp]
        elif val[feature] == prev_state:
            end_state = val[time_stamp]
        elif val[feature] != prev_state:
            end_state = val[time_stamp]
            state_intervals.append((feature, 
                                    prev_state, 
                                    start_state, 
                                    end_state))
            prev_state = val[feature]
            start_state = val[time_stamp]
    state_intervals.append((feature, 
                            prev_state, 
                            start_state, 
                            end_state))
    return state_intervals



