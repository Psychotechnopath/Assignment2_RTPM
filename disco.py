#%%
import pandas as pd

# UD1_df = pd.read_csv('Filtered dataset UD I.csv')
# UD1_df_case_id = list(UD1_df['Case ID'])

# UD2_df = pd.read_csv('UD2 only (filtered).csv')
# UD2_df_case_id = list(UD2_df['Case ID'])

# total_df = pd.read_csv('All cases.csv') #Set CaseId to index col
# total_df_index = total_df.set_index('Case ID', drop=False)

total_df = pd.read_csv('Checkpoint.csv')

#%%
#Only keep last activity
total_df_sub = total_df_index.groupby(total_df_index.index).last()
#Set timestamps as index
total_df_sub.set_index('Complete Timestamp', inplace=True)
#Remove 1 duplicate time value
total_df_sub = total_df_sub.groupby(total_df_sub.index).last()

#Drop unwanted columns
total_df_sub.drop(columns=['Variant', '(case) amount_applied1',
                           '(case) amount_applied2',
                           '(case) amount_applied3',
                           '(case) applicant',
                           '(case) application',
                           '(case) payment_actual1',
                           '(case) payment_actual2',
                           '(case) payment_actual3',
                           '(case) penalty_amount1',
                           '(case) penalty_amount2',
                            '(case) penalty_amount3',
                           '(case) program-id',
                           '(case) year',
                           'docid',
                           'doctype',
                           'lifecycle:transition',
                           'note'], inplace=True)

# If Case ID in UD1_df_case_id -> UD1
# If Case ID in UD2_df_case_id -> UD2
# Else Normal

# 0 = Normal
# 1 = UD1
# 2 = UD2
payment_class = []
for index, value in enumerate(total_df_sub['Case ID']):
    if value in UD1_df_case_id:
        payment_class.append(1)
    elif value in UD2_df_case_id:
        payment_class.append(2)
    else:
        payment_class.append(0)

total_df_sub['payment_class'] = payment_class

total_df_sub[['(case) rejected', 'success', '(case) basic payment',
       '(case) greening', '(case) penalty_ABP', '(case) penalty_AGP', '(case) penalty_AJLP',
       '(case) penalty_AUVP', '(case) penalty_AVBP', '(case) penalty_AVGP',
       '(case) penalty_AVJLP', '(case) penalty_AVUVP', '(case) penalty_B16',
       '(case) penalty_B2', '(case) penalty_B3', '(case) penalty_B4',
       '(case) penalty_B5', '(case) penalty_B5F', '(case) penalty_B6',
       '(case) penalty_BGK', '(case) penalty_BGKV', '(case) penalty_BGP',
       '(case) penalty_C16', '(case) penalty_C4', '(case) penalty_C9',
       '(case) penalty_CC', '(case) penalty_GP1', '(case) penalty_JLP1',
       '(case) penalty_JLP2', '(case) penalty_JLP3', '(case) penalty_JLP5',
       '(case) penalty_JLP6', '(case) penalty_JLP7', '(case) penalty_V5',
        '(case) redistribution', '(case) selected_manually',
       '(case) selected_random', '(case) selected_risk', '(case) small farmer',
       '(case) young farmer']] = \
    total_df_sub[['(case) rejected', 'success',
       '(case) basic payment',
       '(case) greening', '(case) penalty_ABP', '(case) penalty_AGP', '(case) penalty_AJLP',
       '(case) penalty_AUVP', '(case) penalty_AVBP', '(case) penalty_AVGP',
       '(case) penalty_AVJLP', '(case) penalty_AVUVP', '(case) penalty_B16',
       '(case) penalty_B2', '(case) penalty_B3', '(case) penalty_B4',
       '(case) penalty_B5', '(case) penalty_B5F', '(case) penalty_B6',
       '(case) penalty_BGK', '(case) penalty_BGKV', '(case) penalty_BGP',
       '(case) penalty_C16', '(case) penalty_C4', '(case) penalty_C9',
       '(case) penalty_CC', '(case) penalty_GP1', '(case) penalty_JLP1',
       '(case) penalty_JLP2', '(case) penalty_JLP3', '(case) penalty_JLP5',
       '(case) penalty_JLP6', '(case) penalty_JLP7', '(case) penalty_V5',
         '(case) redistribution', '(case) selected_manually',
       '(case) selected_random', '(case) selected_risk', '(case) small farmer',
       '(case) young farmer']].astype(int)

# total_df_sub.to_csv('Checkpoint.csv', header=True)
