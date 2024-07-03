import pandas as pd
import numpy as np
import torch
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from rule_of_thumb import RuleOfThumb

import argparse
import sys

parser = argparse.ArgumentParser(description='Training Rule of Thumb model')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
# parser.add_argument('--help', action='help', help='Show this help message and exit')
args = parser.parse_args()

# if args.help:
#     sys.exit()



df = pd.read_csv('DATA/RoT_inputs.csv')



assert df.isna().sum().sum() == 0


xx = df[df.columns[:-1]].values.astype(float)
yy = df[['prediction']].values.astype(int)

print()
print()
print()
print()

print(f'Args passed to RuleOfThumb: y_outputs=<yy>, x_inputs=<xx>, epochs={args.epochs}, batch_size={args.batch_size}, learning_rate={args.learning_rate}, dropout_rate=0.5')
print()
rot = RuleOfThumb(y_outputs=yy, x_inputs=xx, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, dropout_rate=0.5)
rot_exps = rot.get_explanation(xx)
rot_predictions = rot._explainer_model.predict(torch.from_numpy(xx).to(torch.float)).detach().numpy()


print(f'ROT global importance value is: ' + str(rot._explainer_model.g))

tn, fp, fn, tp = confusion_matrix(yy, rot_predictions).ravel()

print(f'TN rot=0, actual=0: {tn}')
print(f'FP rot=1, actual=0: {fp}')
print(f'FN rot=0, actual=1: {fn}')
print(f'TP rot=1, actual=1: {tp}')
print()
print()
print()
print()
print()

accuracy = accuracy_score(yy, rot_predictions)

print(f'TOTAL accuracy: {accuracy}')

exp_df = pd.DataFrame(rot_exps, columns=df.columns[:771])
new_exp_df = exp_df
new_exp_df.to_csv('DATA/RoT_explanations.csv')

summary_stats = f"""
TN rot=0, actual=0: {tn}
FP rot=1, actual=0: {fp}
FN rot=0, actual=1: {fn}
TP rot=1, actual=1: {tp}
TOTAL accuracy: {accuracy}
"""

with open('DATA/summary_stats.txt', 'w') as f:
    f.write(summary_stats)

# Save predictions to a NumPy file
np.save('DATA/rot_predictions.npy', rot_predictions)

# Pickle the trained PyTorch model
with open('DATA/rot_model.pkl', 'wb') as f:
    pickle.dump(rot, f)

df_classification = pd.read_csv('DATA/classification.csv')
df_extracted = df_classification[['Ground_truth', 'Prediction']]
df_extracted['ROT_Predictions'] = rot_predictions
df_extracted.to_csv('DATA/final_rot.csv', index=False)



# new method start



input_df = pd.read_csv('DATA/classification.csv')



race = [0 if i == 'White' else 1 for i in list(input_df['Race'])]
gender = [0 if i == 'Male' else 1 for i in list(input_df['Gender'])]
party = [0 if i == 'Democratic' else 1 for i in list(input_df['Political_orientation'])]
prediction = [1 if i == 'Yes' else 0 for i in list(input_df['Prediction'])]

for index, summary_text in zip(list(input_df['Unnamed: 0']), list(input_df['Summary'])):

    embedding_df = pd.read_csv(f'DATA/EMBEDDINGS/{index}.csv', index_col=0)
    mean_embedding = list(embedding_df.mean())
    
    embedding_df['race'] = race[index] # irrelevant - will be ignored
    embedding_df['gender'] = gender[index] # irrelevant - will be ignored
    embedding_df['party'] = party[index] # irrelevant - will be ignored
    new_xx = embedding_df.values.astype(float)
    # print(f'{new_xx.shape=}')
    
    new_exps = rot.get_explanation(new_xx)
    
    new_exps_df = pd.DataFrame(new_exps)
    new_exps_df.to_csv(f'DATA/new_exps/{index}.csv')

    
    token_exps = new_exps[:, :-3].sum(axis=1)
    # ignored last 3 columns: those do not correspond to the token embeddings!

    annotate_xx = np.array([mean_embedding + [race[index], 0, 0], mean_embedding + [0, gender[index], 0], mean_embedding + [0, 0, party[index]]])
    annotate_exps = rot.get_explanation(annotate_xx)
    rgp_importances = annotate_exps.sum(axis=1)

    # negative_token_exps = [exp for exp in token_exps if exp < 0]
    # if len(negative_token_exps) > 0:
    #     with open('DATA/direct_rot_stats.txt', 'a') as f:
    #         f.write(f"Index: {index} - Negative token exps: {negative_token_exps}\n")
    #     # print(f"Index: {index} - Negative token exps:", negative_token_exps)
    
    # # print(f'{new_exps.shape=}')
    # # print(f'{token_exps.shape=}')
    
    # with open('DATA/direct_rot_stats.txt', 'a') as f:
    #     f.write(f"Index: {index}\n")
    #     f.write(f"\tMean of token exps: {token_exps.mean()}\n")
    #     new_predictions = rot._explainer_model.predict(torch.from_numpy(new_xx).to(torch.float)).detach().numpy()
    #     f.write(f"\tMean of new_predictions: {new_predictions.mean()}\n")
    #     f.write(f"\tSum of mean (along cols) of new_exps_matrix: {new_exps.mean(axis=0).sum()}\n\n")
        
    new_importance_df = pd.DataFrame(index=list(list(embedding_df.index) + [f'RACE={race[index]}', f'GENDER={gender[index]}', f'PARTY={party[index]}']))
    # print(len(new_importance_df))
    # print(new_importance_df.index)
    new_importance_df['overall_token_importance'] = list(token_exps) + list(rgp_importances)
    
    new_importance_df.to_csv(f'DATA/DIMPS/{index}.csv')



# new method end






def fit_and_print_stats(model, xx, yy, name):
    model.fit(xx, yy.ravel())
    predictions = model.predict(xx)

    tn, fp, fn, tp = confusion_matrix(yy, predictions).ravel()

    print('-------------------------------------------------')
    print(f"{model.__class__.__name__} Model fit with {name} solver:")
    print("Confusion Matrix:")
    print(f'TN rot=0, actual=0: {tn}')
    print(f'FP rot=1, actual=0: {fp}')
    print(f'FN rot=0, actual=1: {fn}')
    print(f'TP rot=1, actual=1: {tp}')

    accuracy = accuracy_score(yy, predictions)
    print("Accuracy:", accuracy)
    print()
    print()
    print()


stats_file = 'DATA/logreg-stats.txt'

def save_stats_to_file(model, name, tn, fp, fn, tp, accuracy):
    with open(stats_file, 'a') as f:
        f.write('-------------------------------------------------\n')
        f.write(f"{model.__class__.__name__} Model fit with {name} solver:\n")
        f.write("Confusion Matrix:\n")
        f.write(f'TN rot=0, actual=0: {tn}\n')
        f.write(f'FP rot=1, actual=0: {fp}\n')
        f.write(f'FN rot=0, actual=1: {fn}\n')
        f.write(f'TP rot=1, actual=1: {tp}\n')
        f.write("Accuracy: {}\n\n\n\n".format(accuracy))

names = ['saga', 'sag', 'newton-cg', 'lbfgs']
models = [
    LogisticRegression(max_iter=10000, random_state=0, solver=s, C=100, tol=0.0000001) for s in names
]

with open(stats_file, 'w') as f:
    f.write('')


for model, name in zip(models, names):
    fit_and_print_stats(model, xx, yy, name)
    predictions = model.predict(xx)
    tn, fp, fn, tp = confusion_matrix(yy, predictions).ravel()
    accuracy = accuracy_score(yy, predictions)
    save_stats_to_file(model, name, tn, fp, fn, tp, accuracy)



print(f'ROT global importance value is: ' + str(rot._explainer_model.g))


