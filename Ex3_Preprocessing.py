#%%
# Imports
import pandas as pd
import datetime as dt

# %%
# Read in checkpoint
total_df = pd.read_csv('Checkpoint.csv')
all_cases_df = pd.read_csv('All cases.csv')

# %%
# activity is same as concept:name
# sum(total_df['activity'] == total_df['concept:name']) == len(total_df['activity'])

# drop duplicate column concept:name
# total_df_filtered = total_df.drop(['concept:name', 'eventid'], axis=1)

# %%
# Sort by time
# total_df_filtered = total_df_filtered.sort_values('Complete Timestamp')

# %%

def conv_datetime(col):
    """
    Convert string to datetime format
    """
    return(dt.datetime.strptime(col, '%Y/%m/%d %H:%M:%S.%f'))

# Convert timestamps to datetime type
all_cases_df['Complete Timestamp'] = all_cases_df['Complete Timestamp'].apply(conv_datetime)
# Get first and last time stamp
all_cases_df_sorted = all_cases_df.sort_values('Complete Timestamp')
last_timestamps = all_cases_df_sorted.groupby('Case ID').last()['Complete Timestamp']
first_timestamps = all_cases_df_sorted.groupby('Case ID').first()['Complete Timestamp']
# Calculate difference
diff_timestamps = last_timestamps - first_timestamps

# %%

# Set case ID as index
total_df = total_df.set_index('Case ID')
# Add 
diff_timestamps_df = diff_timestamps.to_frame()
diff_timestamps_df = diff_timestamps_df.rename(columns={'Complete Timestamp':'Case duration'})
final_df = total_df.join(diff_timestamps_df)

# %%

final_df.to_csv('Ex3_Preprocessing.csv')