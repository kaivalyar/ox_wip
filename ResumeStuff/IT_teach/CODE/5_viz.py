import pandas as pd
import glob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
                aggregate_dict[token] += weight
            else:
                aggregate_dict[token] = weight
                
    return {word: weight for word, weight in aggregate_dict.items() if len(word) > 3}


# Step 2: Create and save the word clouds
def create_and_save_wordclouds(aggregate_dict, pos_output_path, neg_output_path):
    # Separate positive and negative weights for color mapping
    pos_words = {word: weight for word, weight in aggregate_dict.items() if weight > 0}
    neg_words = {word: -weight for word, weight in aggregate_dict.items() if weight < 0}
    
    # Create word clouds
    wordcloud_pos = WordCloud(color_func=lambda *args, **kwargs: "green", width=800, height=400).generate_from_frequencies(pos_words)
    wordcloud_neg = WordCloud(color_func=lambda *args, **kwargs: "red", width=800, height=400).generate_from_frequencies(neg_words)
    
    # Save word clouds to disk
    wordcloud_pos.to_file(pos_output_path)
    wordcloud_neg.to_file(neg_output_path)
    
    # # Display the word clouds
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud_pos, interpolation='bilinear')
    # plt.title('Positive Weights')
    # plt.axis('off')
    # plt.show()
    
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud_neg, interpolation='bilinear')
    # plt.title('Negative Weights')
    # plt.axis('off')
    # plt.show()

# # Step 2: Create a color function for positive and negative weights
# def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
#     if word in word_weights:
#         return 'green' if word_weights[word] > 0 else 'red'
#     return 'black'

# # Step 3: Create the word cloud
# def create_wordcloud(aggregate_dict, output_path='cloud.png'):
#     global word_weights
#     word_weights = aggregate_dict
    
#     wordcloud = WordCloud(color_func=color_func, width=800, height=400, background_color='white').generate_from_frequencies(word_weights)
    
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.savefig(output_path, format='png')
#     plt.close()


# # Step 2: Create the word cloud
# def create_wordcloud(aggregate_dict):
#     # Separate positive and negative weights for color mapping
#     pos_words = {word: weight for word, weight in aggregate_dict.items() if weight > 0}
#     neg_words = {word: -weight for word, weight in aggregate_dict.items() if weight < 0}
    
#     # Create two word clouds for positive and negative words
#     wordcloud_pos = WordCloud(color_func=lambda *args, **kwargs: "green").generate_from_frequencies(pos_words)
#     wordcloud_neg = WordCloud(color_func=lambda *args, **kwargs: "red").generate_from_frequencies(neg_words)
    
#     # Plot the word clouds
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(wordcloud_pos, interpolation='bilinear')
#     plt.title('Positive Weights')
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(wordcloud_neg, interpolation='bilinear')
#     plt.title('Negative Weights')
#     plt.axis('off')
    
#     plt.show()

# Example usage
file_paths = glob.glob('DATA/DeNorms/*.csv')  # Adjust the path to where your CSV files are located
aggregate_dict = aggregate_weights(file_paths)


# Define output file paths
pos_output_path = 'positive_weights_wordcloud.png'
neg_output_path = 'negative_weights_wordcloud.png'

create_and_save_wordclouds(aggregate_dict, pos_output_path, neg_output_path)

# create_wordcloud(aggregate_dict, 'cloud.png')
