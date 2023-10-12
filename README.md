### Introduction

Sepsis is the leading cause of mortality in the United States and the most expensive condition
associated with in-hospital stay, accounting for 6.2% (nearly $24 billion) of total hospital costs.
In particular, Septic shock, the most advanced complication of sepsis due to severe abnormalities
of circulation and/or cellular metabolism, reaches a mortality rate as high as 50% and the annual
incidence keeps rising. It is estimated that as many as 80% of sepsis deaths could be prevented
with early diagnosis and intervention; indeed prior studies have demonstrated that early diagnosis
and treatment of septic shock can significantly decrease patients’ mortality and shorten their length
of stay.

In this project, I have built and compared to machine learning approaches to accurate early diagnosis
of septic shock. The aim is to be able to forecast of the probability that a hospital patient will
enter into shock in the next 24 hours, based on their present charts.

### Top level notebooks:

**split_data.ipynb**: Run once to split data sets into training and test sets. Same splits used in all tests to maintain consistency.

#### RTP

**RTP_experiments.ipynb**: 5 fold cross validation for different parameters/hyperparameters with SVM and Logistic Regression classifiers.

**RTP_evaluate.ipynb**: testing best parameters with SVM and Logisitic Regression agains held out test data.

#### LSTM

**naive_LSTM.ipynb**: No awareness of differing elapsed times between test steps. Vanilla LSTM.

**naive_BiLSTM.ipynb**: Same except with bidirectional LSTM

**time_reconstructed_LSTM.ipynb**: Timelines reconstructed to even 30 minute intervals using Multivariant State Sequences (same preprocessing as for RTP mining). Values are discretized into bins (e.g., VL, L, N, H, VH, by quantiles). Vanilla LSTM.

**time_reconstructed_noencode__LSTM.ipynb**: Same, but without discretization.

**time_reconstructed_BiLSTM.ipynb**: Same, except with bidirectional LSTM. With discretization.

**time_reconstructed_noencode_BiLSTM.ipynb**: Same, but without discretization.

result set	accuracy	precision	recall	f1	auc
1	naive_results	0.891892	0.837209	0.972973	0.900000	0.891892
0	bi_results	0.878378	0.833333	0.945946	0.886076	0.878378
3	tr_noencode_bi_results	0.864865	0.846154	0.891892	0.868421	0.864865
4	tr_noencode_results	0.837838	0.857143	0.810811	0.833333	0.837838
5	tr_results	0.729730	0.717949	0.756757	0.736842	0.729730
2	tr_bi_lstm_results	0.554054	0.529412	0.972973	0.685714	0.554054
