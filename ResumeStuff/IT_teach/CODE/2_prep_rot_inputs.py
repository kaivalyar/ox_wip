import pandas as pd
import numpy as np

input_df = pd.read_csv('DATA/classification.csv')

res = input_df.groupby(['Category', 'Ground_truth', 'Prediction']).count()
res = res[res.columns[:1]]
res.columns = ['COUNT']
res.to_csv('DATA/INPUT_DATA_STATS.csv')
res.to_csv('DATA/input_data_stats.txt', sep="\t")



race = [0 if i == 'White' else 1 for i in list(input_df['Race'])]
gender = [0 if i == 'Male' else 1 for i in list(input_df['Gender'])]
party = [0 if i == 'Democratic' else 1 for i in list(input_df['Political_orientation'])]
prediction = [1 if i == 'Yes' else 0 for i in list(input_df['Prediction'])]

cols = []
all_data = []
nan_count = 0
for index, summary_text in zip(list(input_df['Unnamed: 0']), list(input_df['Summary'])):
    embedding_df = pd.read_csv(f'DATA/EMBEDDINGS/{index}.csv', index_col=0)
    cols = list(embedding_df.columns)
    row = embedding_df.mean().values
    if np.isnan(row.astype(float)).any():
        #print('------')
        print('\t', index)
        #print(row)
        nan_count += 1
        #print('------')
    row = np.append(row, [race[index], gender[index], party[index], prediction[index]])
    all_data.append(row)
    if index % 50 == 0:
        print(f'Prepping RoT inputs {index}')
    
cols.extend(['race','gender','party','prediction'])

all_df = pd.DataFrame(all_data, columns=cols)

all_df.to_csv('DATA/RoT_inputs.csv', index=False)

