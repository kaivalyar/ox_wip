from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

model.eval()




def embed_and_save(input_passage='', output_file_name='empty', append_info=[-1, -1, -1, -1], append_heads=['race', 'gender', 'party', 'prediction']):
    input_passage = input_passage.replace('\n', ' ')
    inputs = tokenizer(input_passage, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    embeddings_list = embeddings[0].cpu().numpy()
    if np.isnan(embeddings_list).any():
        print(f'NANs in {output_file_name}')
    # embeddings_list.extend(append_info)
    
    df = pd.DataFrame(embeddings_list, index=tokens)
    heads = [f'Dim_{i+1}' for i in range(df.shape[1])]
    # heads.extend(append_heads)
    df.columns = heads
    
    df.to_csv(f'DATA/EMBEDDINGS/{output_file_name}.csv', index=True)


input_df = pd.read_csv('DATA/classification.csv')


for index, summary_text in zip(list(input_df['Unnamed: 0']), list(input_df['Summary'])):
    embed_and_save(summary_text, index)
    if index % 50 == 0:
        print(f'embedding generation: iteration {index} of {len(input_df)}')

