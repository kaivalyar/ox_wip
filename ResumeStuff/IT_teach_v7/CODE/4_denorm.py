import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# # # #


imps_df = pd.read_csv(f'DATA/RoT_explanations.csv', index_col=0)
delta = 0

mins = []
maxs = []
ranges = []
sdevs = []
num_tokens = []


for index, imps in enumerate(imps_df.values):
    #print(index)
    #print(imps.shape)
    #print(10_000 * imps)
    embeddings_df = pd.read_csv(f'DATA/EMBEDDINGS/{delta + index}.csv', index_col=0)
    wip_df = embeddings_df.copy(deep=True)
    # embeddings_df = abs(embeddings_df)
    denoms = embeddings_df.sum()
    for col in wip_df.columns:
        #wip_df[col] *= 1000  # no-op, just prevent numbers from getting too small
        wip_df[col] /= denoms[col]
    for i, col in enumerate(wip_df.columns):
        wip_df[col] *= imps[i]
    wip_df['overall_token_importance'] = wip_df.sum(axis=1)
    per_token_importance = wip_df['overall_token_importance']
    per_token_importance = pd.concat([per_token_importance, pd.Series([imps[-3], imps[-2], imps[-1]], index=['RACE', 'GENDER', 'PARTY'], name='overall_token_importance')])
    per_token_importance.to_csv(f'DATA/DeNorms/{index}.csv')
    mins.append(per_token_importance.min())
    maxs.append(per_token_importance.max())
    ranges.append(maxs[-1] - mins[-1])
    sdevs.append(np.std(per_token_importance))
    num_tokens.append(len(per_token_importance))
    if index % 50 == 0:
        print(f'DeNorming iteration {index} of {len(imps_df)}')


