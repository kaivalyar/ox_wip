from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

all_scaled_imps = []
token_importance_list = []

denorm_files = os.listdir('DATA/DeNorms')
for file in denorm_files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join('DATA/DeNorms', file))
        token_importance_list.extend(df['overall_token_importance'].tolist())


positive_tokens = [token for token in token_importance_list if token > 0]
negative_tokens = [token for token in token_importance_list if token < 0]


pos_scaler = MinMaxScaler()
pos_scaler.fit(np.array(positive_tokens).reshape(-1,1))
print(f'POSITIVE STATS: {max(positive_tokens)=} and {min(positive_tokens)=}')
pscaler = max(positive_tokens)

neg_scaler = MinMaxScaler(feature_range=(-1, 0))
neg_scaler.fit(np.array(negative_tokens).reshape(-1,1))
print(f'NEGATIVE STATS: {min(negative_tokens)=} and {max(negative_tokens)=}')
nscaler = abs(min(negative_tokens))

pmaxes, pmins, nmins, nmaxes = [], [], [], []
denorm_files = os.listdir('DATA/DeNorms')
for file in denorm_files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join('DATA/DeNorms', file))
        # df['pos_scaled_importance'] = pos_scaler.transform(np.array(df['overall_token_importance'].values).reshape(-1, 1))
        # df['neg_scaled_importance'] = neg_scaler.transform(np.array(df['overall_token_importance'].values).reshape(-1, 1))
        df['pos_scaled_importance'] = df['overall_token_importance'] / pscaler
        df['neg_scaled_importance'] =  df['overall_token_importance'] / nscaler
        
        df['is_neg'] = df['overall_token_importance'] < 0
        df['is_pos'] = df['overall_token_importance'] > 0
        
        df['pos_scaled_importance'] *= df['is_pos']
        df['neg_scaled_importance'] *= df['is_neg']

        pmaxes.append(max(df['pos_scaled_importance']))
        pmins.append(min(df['pos_scaled_importance']))
        nmins.append(min(df['neg_scaled_importance']))
        nmaxes.append(max(df['neg_scaled_importance']))

        df['scaled_token_importance'] = df['pos_scaled_importance'] + df['neg_scaled_importance']
        df['abs_scaled_token_importance'] = abs(df['scaled_token_importance'])

        # df['verify'] = abs(df['pos_scaled_importance']) + abs(df['neg_scaled_importance']) - df['abs_scaled_token_importance']
        # print(sum(df['verify']))
        

        
        new_df = df[['Unnamed: 0', 'scaled_token_importance']]
        
        if False:
            new_df.loc[:, 'scaled_token_importance'] = np.cbrt(new_df['scaled_token_importance'])
        
        # new_df['scaled_token_importance'] = np.cbrt(new_df['scaled_token_importance'])

        # new_df['scaled_token_importance'] = new_df['scaled_token_importance'] ** (1/3)
        
        # new_df['scaled_token_importance'] = new_df['scaled_token_importance'] ** (1/27)
        new_df.to_csv(f'DATA/scaled/{file}', index=False)
        all_scaled_imps.extend(new_df['scaled_token_importance'].tolist())

        # df['scaled_token_importance'] = df['overall_token_importance'].apply(lambda x: pos_scaler.transform([[x]])[0][0] if x > 0 else (neg_scaler.transform([[x]])[0][0] if x < 0 else x)

print(f'After scaling, the positive importances begin at: {min(pmins)} to {max(pmins)}')
print(f'After scaling, the positive importances go up to: {min(pmaxes)} to {max(pmaxes)}')
print()
print(f'After scaling, the negative importances begin at: {max(nmaxes)} to {min(nmaxes)}')
print(f'After scaling, the negative importances go up to: {max(nmins)} to {min(nmins)}')

# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))
sns.displot(all_scaled_imps)
plt.title('Distribution of Token Importances')
plt.xlabel('importance')
plt.ylabel('Frequency')
# plt.yscale('symlog')
plt.savefig('RoT_plot.png')
plt.close()

print(f'The minimum value of all_scaled_imps is: {min(all_scaled_imps)}')
print(f'The maximum value of all_scaled_imps is: {max(all_scaled_imps)}')
