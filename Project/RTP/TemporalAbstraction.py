import pandas as pd
import numpy as np

from RTP.Config import Options as opts


class State:
	def __init__(self, feature, value):
		self.feature = feature
		self.value = value
	def __str__(self):
		return f'({self.feature}, {self.value})'
	def __eq__(self, other):
		if self.feature == other.feature and self.value == other.value:
			return True
		return False
	def __hash__(self):
		return hash((self.feature,self.value))

class StateInterval:
	def __init__(self, feature, value, start, end):
		self.feature = feature
		self.value = value
		self.start = start
		self.end = end
	def __gt__(self, state2):
		return self.start > state2.start
	def __str__(self):
		return f'({self.feature}, {self.value}, {self.start}, {self.end})'
	def find_relation(self, s2):
		if self.end < s2.start:
			return 'b'
		if self.start <= s2.start <= self.end:
			return 'c'
		

def discretization(features_1, features_0, temperature=False):
    if features_0 is None:
        values = features_1
    else:
        values = pd.concat([features_1, features_0])

    if temperature:
        bins = [0, 36.1, 37.3, values.max() + 1]
        labels = ['L', 'N', 'H']
    else:
        percentiles = [10, 25, 75, 90, 100]
        labels = ['VL', 'L', 'N', 'H', 'VH']
        bins = [0] + [np.percentile(values, p) for p in percentiles]
        bins[-1] = values.max() + 1

    f = lambda x: labels[i] if not isinstance(x, str) and \
                  bins[i] <= x < bins[i+1] else x

    for i in range(len(bins) - 1):
        features_1 = features_1.apply(f)
        if features_0 is not None:
            features_0 = features_0.apply(f)
    return features_1, features_0
    
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
            state_intervals.append(StateInterval(feature, 
                                                 prev_state, 
                                                 start_state, 
                                                 end_state))
            prev_state = val[feature]
            start_state = val[time_stamp]
            end_state = val[time_stamp]
    state_intervals.append(StateInterval(feature, 
                                         prev_state, 
                                         start_state, 
                                         end_state))
    return state_intervals

def MultiStateSequence(data, features, time_stamp):
    MSS = []
    for feature in features:
        MSS.extend(state_generation(data, feature, time_stamp))
    MSS.sort(key=lambda x: x.start)
    return MSS

def find_state_matches(mss, state, fi):
	'''
	Given an MSS, find the index of state intervals with the same feature and value as of state, starting from fi index
	'''
	match = []
	for i in range (fi, len(mss)):
		if state.feature == mss[i].feature and state.value == mss[i].value:
			match.append(i)
	return match

def MSS_contains_Pattern(mss, p, i, fi, prev_match):
	'''
	A recursive function that determines whether a sequence contains a pattern or not, based 
	on DEFINITION 2 from Batal et al.
	'''			
	if i >= len(p.states):
		return True, prev_match
	same_state_index = find_state_matches(mss, p.states[i], fi)
	for fi in same_state_index:
		flag = True
		for pv in range(0,len(prev_match)):
			if prev_match[pv].find_relation(mss[fi]) != p.relation[pv][i]:
				flag = False
				break
		if flag:
			prev_match.append(mss[fi])
			contains, seq = MSS_contains_Pattern(mss, p, i+1, 0, prev_match)
			if contains:
				return True, seq
			else:
				del prev_match[-1]
	return False, np.nan

def recent_state_interval(mss, j, g):
	'''
	Determines whether a state interval is recent or not, based on DEFINITION 3 from Batal et al.
	'''
	if mss[len(mss)-1].end - mss[j].end <= g:
		return True
	flag = False
	for k in range(j+1, len(mss)):
		if mss[j].feature == mss[k].feature:
			flag = True
	if not flag:
		return True
	return False

def get_index_in_sequence(mss, e):
	'''
	Finds the index of a state interval in an MSS
	'''
	for i in range(0,len(mss)):
		if mss[i] == e:
			return i
	return -1

def sequences_containing_state(RTPlist, new_s):
	'''
	Given a new state and a list of MSS's, determines which contain this state
	'''
	p_RTPlist = []
	for z in RTPlist:
		for e in z:
			if e.feature == new_s.feature and e.value == new_s.value:
				p_RTPlist.append(z)
				break
	return p_RTPlist

def find_all_frequent_states(D, support):
	'''
	Given all MSS's of a group and their minimum support, finds all frequent states
	'''
	freq_states = []
	for f in opts.numerical_feat:
		for v in opts.num_categories:
			state = State(f, v)
			if len(sequences_containing_state(D, state)) >= support:
				freq_states.append(state)
	return freq_states