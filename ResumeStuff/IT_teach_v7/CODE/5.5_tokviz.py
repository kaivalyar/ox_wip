



import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read and aggregate the CSV files
def aggregate_weights(file_paths):
    aggregate_dict = {}

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            # print(row)
            token = row['Unnamed: 0']
            weight = row['overall_token_importance']
            if token in aggregate_dict:
                aggregate_dict[token].append(weight)
            else:
                aggregate_dict[token] = [weight]
                
    # return {word: weight for word, weight in aggregate_dict.items() if len(word) > 0}
    
    # return {word: weight for word, weight in aggregate_dict.items() if (len(word) > 3) and (word not in ['[CLS]', '[SEP]'])}
    return {word: weight for word, weight in aggregate_dict.items()}
    

stencil = 'DATA/DIMPS/{}.csv'
rimps = pd.read_csv('DATA/final_rot.csv')

negs = sorted(list(rimps[rimps['ROT_Predictions'] == 0].index))
poss = sorted(list(rimps[rimps['ROT_Predictions'] == 1].index))

tok2list_n = aggregate_weights([stencil.format(fname) for fname in negs])
tok2list_p = aggregate_weights([stencil.format(fname) for fname in poss])

result_dict = {}
for word, weight in tok2list_n.items():
    result_dict[word] = {'negs': weight, 'poss': []}

for word, weight in tok2list_p.items():
    if word in result_dict:
        result_dict[word]['poss'] = weight
    else:
        result_dict[word] = {'negs': [], 'poss': weight}
    
result_list = []
for word in result_dict:
    result_list.append({'token': word, 'pos_weights': result_dict[word]['poss'], 'neg_weights': result_dict[word]['negs']})


result_df = pd.DataFrame(result_list) # , columns=['token', 'pos_weights', 'neg_weights'])

for index, row in result_df.iterrows():
    token = row['token']
    pos_data = row['pos_weights']
    neg_data = row['neg_weights']
    
    plt.figure(figsize=(10, 6))
    overall_total = sum(pos_data) + sum(neg_data)
    plt.suptitle(f'Token Importance Distributions for "{token}" (overall  sum: {overall_total:.2f})\n\n', fontsize=14)
    
    plt.subplot(1, 2, 2)
    if len(pos_data) > 0:
        mean_pos = sum(pos_data) / len(pos_data)
        min_pos = min(pos_data)
        max_pos = max(pos_data)
        sns.histplot(pos_data, kde=True, color='green')
        plt.title(f'Total "{len(pos_data)}" values (RoT-pred = 1)\n{min_pos:.2f} to {max_pos:.2f}, mean {mean_pos:.2f}')
        plt.xlabel('Weights')
        plt.ylabel('Frequency')
        plt.axvline(mean_pos, color='black', linestyle='dashed', linewidth=1, label=f'Mean: {mean_pos:.2f}')
        plt.axvline(min_pos, color='black', linestyle='dashed', linewidth=1,  label=f'Mean: {min_pos:.2f}')
        plt.axvline(max_pos, color='black', linestyle='dashed', linewidth=1,  label=f'Mean: {max_pos:.2f}')
    
    else:
        plt.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        
    plt.subplot(1, 2, 1)
    if len(neg_data) > 0:
        mean_pos = sum(neg_data) / len(neg_data)
        min_pos = min(neg_data)
        max_pos = max(neg_data)
        sns.histplot(neg_data, kde=True, color='red')
        plt.title(f'Total "{len(neg_data)}" values (RoT-pred = 0)\n{min_pos:.2f} to {max_pos:.2f}, mean {mean_pos:.2f}')
        plt.xlabel('Weights')
        plt.ylabel('Frequency')
        plt.axvline(mean_pos, color='black', linestyle='dashed', linewidth=1, label=f'Mean: {mean_pos:.2f}')
        plt.axvline(min_pos, color='black', linestyle='dashed', linewidth=1, label=f'Mean: {min_pos:.2f}')
        plt.axvline(max_pos, color='black', linestyle='dashed', linewidth=1, label=f'Mean: {max_pos:.2f}')
    else:
        plt.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
         
    plt.savefig(f'DATA/distplots/{token}.png')
    plt.close()


result_df.to_csv('DATA/tokens.csv', index=False)









# all_files = [f for f in glob.glob('DATA/DIMPS/*.csv')]

# df_final_rot = pd.read_csv('DATA/final_rot.csv')

# file_paths_rot_0 = sorted([fname for fname in all_files if df_final_rot['ROT_Predictions'].iloc[int(fname.split('/')[-1].split('.')[0])-2] == 0])
# file_paths_rot_1 = sorted([fname for fname in all_files if df_final_rot['ROT_Predictions'].iloc[int(fname.split('/')[-1].split('.')[0])-2] == 1])

# print(f'{file_paths_rot_0=} \n\n\n {file_paths_rot_1=}')
# print(f'{len(file_paths_rot_0)=} \n\n\n {len(file_paths_rot_1)=}')


