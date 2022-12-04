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