import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score

class Logging:
    def __init__(self, file_path):
        self.file_path = file_path

    def info(self, message):
        with open(self.file_path, 'a') as FP:
            FP.write(f'{message}\n')


def train_test_split(df, sample_id, test_size, random_state=42):
    np.random.seed(random_state)
    unique_ids = df[sample_id].unique()
    n_ids = len(unique_ids)
    test_ids = np.random.choice(unique_ids, int(n_ids * test_size))
    df_test = df[df[sample_id].isin(test_ids)]
    df_train = df[~df[sample_id].isin(test_ids)]
    return df_train, df_test
    

def evaluate(test_labels, pred_labels):
    metrics = {}
    metrics['accuracy'] = accuracy_score(test_labels, pred_labels)
    metrics['precision'] = precision_score(test_labels, pred_labels)
    metrics['recall'] = recall_score(test_labels, pred_labels)
    metrics['f1'] = f1_score(test_labels, pred_labels)
    metrics['auc'] = roc_auc_score(test_labels, pred_labels)
    return metrics

