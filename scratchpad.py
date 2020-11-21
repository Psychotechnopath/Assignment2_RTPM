for index, row in total_df_sub.iterrows():
    if row['(case) greening']:
        application_types.append(1)
    if row['(case) redistribution']:
        application_types.append(2)
    if row['(case) small farmer']:
        application_types.append(3)
    if row['(case) young farmer']:
        application_types.append(4)
    if row['(case) basic payment']:
        application_types.append(5)
    if row['(case) selected_random']:
        selection_types.append(1)
    if row['(case) selected_risk']:
        selection_types.append(2)
    if row['(case) selected_manually']:
        selection_types.append(3)
    if row['(case) penalty_ABP']:
        penalty_types.append(1)
        continue
    if row['(case) penalty_V5']:
        penalty_types.append(2)
        continue
    if row['(case) penalty_AGP']:
        penalty_types.append(3)
        continue
    if row['(case) penalty_AJLP']:
        penalty_types.append(4)
        continue
    if row['(case) penalty_AUVP']:
        penalty_types.append(5)
        continue
    if row['(case) penalty_AVBP']:
        penalty_types.append(6)
        continue
    if row['(case) penalty_AVGP']:
        penalty_types.append(7)
        continue
    if row['(case) penalty_AVJLP']:
        penalty_types.append(8)
        continue
    if row['(case) penalty_AVUVP']:
        penalty_types.append(9)
        continue
    if row['(case) penalty_B16']:
        penalty_types.append(10)
        continue
    if row['(case) penalty_B2']:
        penalty_types.append(11)
        continue
    if row['(case) penalty_B3']:
        penalty_types.append(12)
        continue
    if row['(case) penalty_B4']:
        penalty_types.append(13)
        continue
    if row['(case) penalty_B5']:
        penalty_types.append(14)
        continue
    if row['(case) penalty_B5F']:
        penalty_types.append(15)
        continue
    if row['(case) penalty_B6']:
        penalty_types.append(16)
        continue
    if row['(case) penalty_BGK']:
        penalty_types.append(17)
        continue
    if row['(case) penalty_BGKV']:
        penalty_types.append(18)
        continue
    if row['(case) penalty_BGP']:
        penalty_types.append(19)
        continue
    if row['(case) penalty_C16']:
        penalty_types.append(20)
        continue
    if row['(case) penalty_C4']:
        penalty_types.append(21)
        continue
    if row['(case) penalty_C9']:
        penalty_types.append(22)
        continue
    if row['(case) penalty_CC']:
        penalty_types.append(23)
        continue
    if row['(case) penalty_GP1']:
        penalty_types.append(24)
        continue
    if row['(case) penalty_JLP1']:
        penalty_types.append(25)
        continue
    if row['(case) penalty_JLP2']:
        penalty_types.append(26)
        continue
    if row['(case) penalty_JLP3']:
        penalty_types.append(27)
        continue
    if row['(case) penalty_JLP5']:
        penalty_types.append(28)
        continue
    if row['(case) penalty_JLP6']:
        penalty_types.append(29)
        continue
    if row['(case) penalty_JLP7']:
        penalty_types.append(30)
        continue
    else:
        penalty_types.append(0)
        selection_types.append(0)
        application_types.append(0)

##SMall check to see if i can push again from laptop