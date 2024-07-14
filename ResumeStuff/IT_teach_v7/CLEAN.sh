
mv DATA/classification.csv classification.csv
rm -rf DATA/*
mv classification.csv DATA/classification.csv

mkdir DATA/EMBEDDINGS
mkdir DATA/new_exps
mkdir DATA/DeNorms
mkdir DATA/scaled
mkdir DATA/tokens
mkdir DATA/DIMPS
mkdir DATA/distplots
touch DATA/direct_rot_stats.txt


rm positive_weights_wordcloud.png
rm negative_weights_wordcloud.png
rm rot_plot.png
