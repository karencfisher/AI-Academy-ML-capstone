class Options:
    # experiment settings
    early_prediction = 24      # For'left' alignment, early_prediction = the time window starting from the begnning 
    observation_window = 7*24  # *** Check the length of trajectories
    alignment = 'right' 
    settings = 'trunc'
    interval = 30
    
    numerical_feat = ['SystolicBP','HeartRate','RespiratoryRate','Temperature', 'WBC']
    num_categories = ['VL', 'L', 'N', 'H', 'VH']
    timestamp_variable = 'MinutesFromArrival'
