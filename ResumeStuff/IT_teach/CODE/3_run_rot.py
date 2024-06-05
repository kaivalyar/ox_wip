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
    LogisticRegression(max_iter=100000, random_state=0, solver=s, C=100, tol=0.0000001) for s in names
]

with open(stats_file, 'w') as f:
    f.write('')


for model, name in zip(models, names):
    fit_and_print_stats(model, xx, yy, name)
    predictions = model.predict(xx)
    tn, fp, fn, tp = confusion_matrix(yy, predictions).ravel()
    accuracy = accuracy_score(yy, predictions)
    save_stats_to_file(model, name, tn, fp, fn, tp, accuracy)






