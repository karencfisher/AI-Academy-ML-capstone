import numpy as np
from tqdm import tqdm

import RTP.RTPmining as rm
import RTP.TemporalAbstraction as ta
from RTP.Config import Options as opts
from tools.tools import evaluate, Logging


class RTPclf:
    def __init__(self, model, RTM_params, logfile):
        self.min_support_1 = RTM_params['min_support_pos']
        self.min_support_0 = RTM_params['min_support_neg']
        self.max_gap = RTM_params['max_gap']
        self.model = model
        self.patterns = None

        self.logger = Logging(logfile)
        self.logger.info('----------------------------------------')
        self.logger.info(f'Model: {self.model}')
        self.logger.info(f'RTM_params: {RTM_params}')       

    def __create_patterns(self, MSS_pos, MSS_neg):
        n1 = len(MSS_pos)
        n0 = len(MSS_neg)

        pos_patterns = rm.pattern_mining(MSS_pos, 
                                  self.max_gap, 
                                  self.min_support_1*n1,
                                  self.logger)
        neg_patterns = rm.pattern_mining(MSS_neg, 
                                  self.max_gap, 
                                  self.min_support_0*n0,
                                  self.logger)

        all_patterns = list(pos_patterns)
        for j in range(0, len(neg_patterns)):
            if not any((x == neg_patterns[j]) for x in all_patterns):
                all_patterns.append(neg_patterns[j])
        return all_patterns

    def __create_binary_matrix(self, data):
        binary_matrix = np.zeros((len(data), len(self.patterns)))
        for i in range(len(data)):
            for j in range(len(self.patterns)):
                    present = rm.recent_temporal_pattern(data[i], 
                                                         self.patterns[j], 
                                                         self.max_gap)
                    if(present):
                        binary_matrix[i,j] = 1
                    else:
                        binary_matrix[i,j] = 0
        return binary_matrix

    def trainCV(self, MSS_pos, MSS_neg, num_folds, verbose=True):
        train_data = list(MSS_pos)
        train_data.extend(MSS_neg)
        train_labels = list(np.ones(len(MSS_pos)))
        train_labels.extend(np.zeros(len(MSS_neg)))

        if verbose:
            print('Generating binary matrix, this may take several minutes...')
        self.patterns = self.__create_patterns(MSS_pos, MSS_neg)
        train_matrix = self.__create_binary_matrix(train_data).tolist()
        if verbose:
            print("Whew, that was hard!")
 
        subset_size = len(train_matrix) // num_folds
        train_scores = []
        val_scores = []
        val_labels = []
        pred_labels = []
        message = f'Beginning training SVM with {num_folds}-fold validation...'
        self.logger.info(message)
        if verbose:
            print(message)
        for i in range(num_folds):
            trainMatrix = train_matrix[:i*subset_size] + train_matrix[(i+1)*subset_size:]
            valMatrix = train_matrix[i*subset_size:(i+1)*subset_size]

            trainLabels = train_labels[:i*subset_size]  + train_labels[(i+1)*subset_size:]
            valLabels = train_labels[i*subset_size:(i+1)*subset_size]
            val_labels.extend(valLabels)
            
            self.model.fit(trainMatrix, trainLabels)
            train_accuracy = self.model.score(trainMatrix, trainLabels)
            train_scores.append(train_accuracy)
            val_accuracy = self.model.score(valMatrix, valLabels)
            val_scores.append(val_accuracy)
            pred_labels.extend(self.model.predict(valMatrix))

            message = (f'Fold {i + 1}: train_accuracy = {train_accuracy} val_accuracy = {val_accuracy}')
            if verbose:
                print(message)
            self.logger.info(message)
        
        if verbose:
            print('Training completed!')
        self.logger.info('Training completed!')
        metrics = evaluate(val_labels, pred_labels)
        self.logger.info(f'Final metrics: {metrics}')
        return metrics

    def evaluate(self, MSS_pos, MSS_neg, verbose=True):
        '''
        Evaluate the model on new data
        '''
        assert self.patterns is not None, 'model has not been trained yet'

        test_data = list(MSS_pos)
        test_data.extend(MSS_neg)
        test_labels = list(np.ones(len(MSS_pos)))
        test_labels.extend(np.zeros(len(MSS_neg)))

        if verbose:
            print('Generating binary matrix may take a moment...')
        matrix = self.__create_binary_matrix(test_data).tolist()
        if verbose:
            print('Predicting outcomes...')
        pred_labels = self.model.predict(matrix)
        if verbose:
            print('Done!')
        metrics = evaluate(test_labels, pred_labels)
        return pred_labels, metrics


def make_MSS(events):
    assert len(events) > 0, 'empty sequence'
    MSS = []
    VisitIDs = events['VisitIdentifier'].unique()
    for visit_id in tqdm(VisitIDs):
        data = events[events['VisitIdentifier'] == visit_id]
        mss = ta.MultiStateSequence(data, opts.numerical_feat, 'MinutesFromArrival')
        MSS.append(mss)
    return MSS

def preprocess(pos_events, neg_events):
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

    for value in opts.numerical_feat:
        temperature = value == 'Temperature'
        pos_events[value], neg_events[value] = ta.discretization(pos_events[value], 
                                                                 neg_events[value],
                                                                 temperature=temperature) 
    MSS_pos = make_MSS(pos_events)
    MSS_neg = make_MSS(neg_events)
    return MSS_pos, MSS_neg