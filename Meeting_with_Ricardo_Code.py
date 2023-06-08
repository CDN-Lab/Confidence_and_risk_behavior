#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 21:02:27 2023

@author: raymondcn
"""

import pandas as pd
import glob
import os
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

root_path = '/Volumes/UCDN/datasets/IDM'
save_dir = os.path.join(root_path, 'confidence_scores')

# Define the file path
file_paths = glob.glob(os.path.join(root_path, 'raw_csv', '*.csv'))

# Define an empty dataframe 
averages_df = pd.DataFrame(columns=['File', 'Confidence Average Block 1', 'Confidence Average Block 2', 'Confidence Average Block 3', 'Confidence Average Block 4', 'Risk Amount', "All Average", "Count", "Ambiguity Amount", "Risk Amount All"])

# Loop through each participant CSV file
for file_path in file_paths:

    df = pd.read_csv(file_path)
    
    # Filter the dataframe to include only low_vol_low_risk
    block1_df = df[df['cpdm_run_dimension'] == 'low_vol_low_risk']
    
    # Filter the dataframe to include only low_vol_high_risk
    block2_df = df[df['cpdm_run_dimension'] == 'low_vol_high_risk']
    
    # Filter the dataframe to include only high_vol_low_risk
    block3_df = df[df['cpdm_run_dimension'] == 'high_vol_low_risk']
    
    # Filter the dataframe to include only high_vol_high_risk
    block4_df = df[df['cpdm_run_dimension'] == 'high_vol_high_risk']
    
    # Filter for ambiguity level 0 
    ambiguity_level = df[df['crdm_amb_lev'] > 0]
    
    # Filter for ambiguity level 0 
    risk_level = df[df['crdm_amb_lev'] == 0]
    
    # Calculate the average of column 'cpdm_conf' for block 1
    block1_average = block1_df['cpdm_conf'].mean()
    
    # Calculate the average of column 'cpdm_conf' for block 2
    block2_average = block2_df['cpdm_conf'].mean()
    
    # Calculate the average of column 'cpdm_conf' for block 3
    block3_average = block3_df['cpdm_conf'].mean()
    
    # Calculate the average of column 'cpdm_conf' for block 4
    block4_average = block4_df['cpdm_conf'].mean()
    
    ambiguity_choice = ambiguity_level['crdm_choice'].mean()
    
    risk_choice = risk_level['crdm_choice'].mean()
    
    risk_choice_all = df['crdm_choice'].mean()
    
    all_average = df['cpdm_conf'].mean()
    
    count = df['cpdm_conf'].sum()
    
    # Extract the file name from the file path
    file_name = os.path.basename(file_path).replace('.csv', '')
    
    # Add the file name and averages to the  dataframe
    averages_df = averages_df.append({'File': file_name, 'Confidence Average Block 1': block1_average, 
                                      'Confidence Average Block 2': block2_average, 
                                      'Confidence Average Block 3': block3_average, 
                                      'Confidence Average Block 4': block4_average, 
                                      'Ambiguity Amount': ambiguity_choice, 
                                      'Risk Amount': risk_choice, 
                                      'Risk Amount All': risk_choice_all,
                                      'Count': count, 
                                      'All Average': all_average},
                                     ignore_index=True)

# Calculate the low confidence average
averages_df['Confidence Average Low'] = (averages_df['Confidence Average Block 1'] + averages_df['Confidence Average Block 3']) / 2


# Calculate the high confidence average
averages_df['Confidence Average High'] = (averages_df['Confidence Average Block 2'] + averages_df['Confidence Average Block 4']) / 2


# Save the averages dataframe to a CSV file
averages_df.to_csv(os.path.join(save_dir, 'conf_averages.csv'), index=False)

# Compute difference 

averages_df['Difference'] = averages_df['Confidence Average High'] - averages_df['Confidence Average Low']


average_df = averages_df.rename(columns={'File': 'subject'})

# Load Dataframe

root_dir = '/Volumes/UCDN/datasets/IDM'
df_aggregate = pd.read_csv(os.path.join(root_dir,"Aggregate_All_Model_Parameters.csv"))
print(list(df_aggregate))
['subject', 'cpdm_conf_crit_b1', 'cpdm_conf_crit_b2','cpdm_conf_crit_b3', 'cpdm_conf_crit_b4', 'crdm_alpha']

# df_aggregate = df_aggregate[df_aggregate['cpdm_confidence_flag'] == 0]


# Compute the mean for the confidence criteria in each block 

block1 = 'cpdm_conf_crit_b1'
block3 = 'cpdm_conf_crit_b3'
block2 = 'cpdm_conf_crit_b2'
block4 = 'cpdm_conf_crit_b4'


block1_mean = df_aggregate['cpdm_conf_crit_b1'].mean()
block2_mean = df_aggregate['cpdm_conf_crit_b2'].mean()
block3_mean = df_aggregate['cpdm_conf_crit_b3'].mean()
block4_mean = df_aggregate['cpdm_conf_crit_b4'].mean()

# Compute average for high incentive blocks

block2 = 'cpdm_conf_crit_b2'
block4 = 'cpdm_conf_crit_b4'
high_incentive_confidence = 'high_incentives_conf_scores'

df_aggregate[high_incentive_confidence] = (df_aggregate[block2] + df_aggregate[block4]) / 2
print(df_aggregate[high_incentive_confidence])

high_incentive_confidence_mean = df_aggregate['high_incentives_conf_scores'].mean()

# Compute average for low incentive blocks

block1 = 'cpdm_conf_crit_b1'
block3 = 'cpdm_conf_crit_b3'
low_incentive_confidence = 'low_incentives_conf_scores'

df_aggregate[low_incentive_confidence] = (df_aggregate[block1] + df_aggregate[block3]) / 2
print(df_aggregate[low_incentive_confidence])

low_incentive_confidence_mean = df_aggregate['low_incentives_conf_scores'].mean()

# Compute difference between high and low incentive confidence scores

high_incentive_confidence = 'high_incentives_conf_scores'
low_incentive_confidence = 'low_incentives_conf_scores'
incentive_conf_difference = 'incentives_confidence_difference'

df_aggregate[incentive_conf_difference] = df_aggregate[high_incentive_confidence] - df_aggregate[low_incentive_confidence]

incentive_difference_confidence_mean = df_aggregate['incentives_confidence_difference'].mean()


# Merge or join the datasets based on subject ID
merged_data = pd.merge(df_aggregate, average_df, on='subject')

print(merged_data.shape)

merged_data_filtered = merged_data[merged_data['cpdm_confidence_flag'] == 0]

print(merged_data_filtered.shape)

merged_data_filtered = merged_data_filtered[merged_data_filtered['crdm_R2'] > 0.5]

print(merged_data_filtered.shape)

merged_data_filtered = merged_data_filtered[merged_data_filtered['crdm_prob_span'] > 0.95]

print(merged_data_filtered.shape)


x1 = merged_data_filtered['Difference']
y1 = merged_data_filtered['Risk Amount']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('Difference between % Choice', fontsize=15)
plt.ylabel('% Risky Choice', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data_filtered['incentives_confidence_difference']
y1 = merged_data_filtered['crdm_alpha']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('Conf Crit Difference', fontsize=15)
plt.ylabel('Alpha', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


x1 = merged_data_filtered['Confidence Average High']
y1 = merged_data_filtered['Risk Amount']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('Confidence Average High', fontsize=15)
plt.ylabel('% Risk', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data['Confidence Average High']
y1 = merged_data['Risk Amount']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('Confidence Average High', fontsize=15)
plt.ylabel('% Risk', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data_filtered['high_incentives_conf_scores']
y1 = merged_data_filtered['crdm_alpha']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()









































## TESTS

# Low incentive sanity check

x1 = merged_data['Confidence Average Low']
y1 = merged_data['low_incentives_conf_scores']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)


a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='mediumaquamarine', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% High Confidence in Low Incentive Blocks ', fontsize=15)
plt.ylabel('Confidence Criteria in Low Incentive Blocks ', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

# High incentive sanity check

x1 = merged_data['Confidence Average High']
y1 = merged_data['high_incentives_conf_scores']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)


a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='red', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% High Confidence in High Incentive Blocks ', fontsize=15)
plt.ylabel('Confidence Criteria in High Incentive Blocks ', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

# Difference sanity check

x1 = merged_data['Difference']
y1 = merged_data['incentives_confidence_difference']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)


a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% High Confidence Incentive Difference', fontsize=15)
plt.ylabel('Confidence Criteria Difference ', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


# Risk sanity check 

x1 = merged_data['Risk Amount']
y1 = merged_data['crdm_alpha']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)


a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='orange', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Risky Option', fontsize=15)
plt.ylabel('Alpha', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


### Tests within % choice


x1 = merged_data['Confidence Average Low']
y1 = merged_data['Confidence Average High']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='yellow', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Confidence High Incentive', fontsize=15)
plt.ylabel('% Confindence Low Incentive', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


x1 = merged_data['high_incentives_conf_scores']
y1 = merged_data['low_incentives_conf_scores']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='purple', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Confidence High Conf Crit', fontsize=15)
plt.ylabel('% Confindence Low Conf Crit', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data['Confidence Average Low']
y1 = merged_data['Risk Amount']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='blue', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Confidence High Low Incentive', fontsize=15)
plt.ylabel('% Risky Choice', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data['Confidence Average Low']
y1 = merged_data['crdm_alpha']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='blue', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Confidence High Low Incentive', fontsize=15)
plt.ylabel('% Risky Choice', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

alpha = df_aggregate['crdm_alpha']

log_alpha = np.log10(alpha)

x1 = averages_df['Confidence Average Low']
y1 = log_alpha

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)


x1 = merged_data['Confidence Average High']
y1 = merged_data['Risk Amount']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Confidence High High Incentive', fontsize=15)
plt.ylabel('% Risky Choice', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data['Confidence Average High']
y1 = merged_data['crdm_alpha']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Confidence High High Incentive', fontsize=15)
plt.ylabel('crdm_alpha', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data['Confidence Average High']
y1 = log_alpha

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Confidence High High Incentive', fontsize=15)
plt.ylabel('% Risky Choice', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


x1 = merged_data['Difference']
y1 = merged_data['Risk Amount']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Difference', fontsize=15)
plt.ylabel('% Risky Choice', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data['Difference']
y1 = log_alpha

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('% Difference', fontsize=15)
plt.ylabel('Alpha', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


x1 = merged_data['incentives_confidence_difference']
y1 = merged_data['Risk Amount']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('Parameter Difference', fontsize=15)
plt.ylabel('% Risky Choice', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

x1 = merged_data['incentives_confidence_difference']
y1 = merged_data['crdm_alpha']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('Parameter Difference', fontsize=15)
plt.ylabel('Alpha', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

y1 = merged_data['Risk Amount All']
x1 = merged_data['crdm_alpha']

correlation, p_value = spearmanr(x1, y1)
print(correlation, p_value)

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1, color='green', edgecolors='gray', s=65)
plt.plot(x1, a*x1+b, 'r-', color='gray')
plt.xlabel('Risk Amount All', fontsize=15)
plt.ylabel('Alpha', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()










