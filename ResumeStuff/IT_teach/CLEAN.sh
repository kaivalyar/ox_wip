
mv DATA/classification.csv classification.csv
rm -rf DATA/*
mv classification.csv DATA/classification.csv

mkdir DATA/EMBEDDINGS
mkdir DATA/DeNorms

rm positive_weights_wordcloud.png
rm negative_weights_wordcloud.png
