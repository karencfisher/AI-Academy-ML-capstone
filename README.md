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
