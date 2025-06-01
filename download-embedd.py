# Downloading and embedding data

# Import data

import kagglehub
from kagglehub import KaggleDatasetAdapter

# importing libraries

import pandas as pd
import numpy as np

# For BERT
import random
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Cleaning text
from bs4 import BeautifulSoup
import re

def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to get sentence embeddings. See:
    https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # Sum columns
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Set the path to the file you'd like to load
file_path = "sp500_headlines_2008_2024.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

# First, clean text
df['TitleClean'] = df['Title'].apply(text_cleaning)

# Input text
text_list = df['TitleClean'].to_list()

# Load BERT tokenizer and model
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoded_input = [tokenizer(expr, return_tensors="pt") for expr in text_list]

# Create word embeddings
print('Creating word embeddings')
sentence_embeddings = []
for i in range(len(encoded_input)) :
    print(str(i+1)+'/'+str(len(encoded_input)))
    model_output = model(**encoded_input[i])
    sentence_embeddings.append(mean_pooling(model_output, encoded_input[i]['attention_mask']).detach().numpy()[0])
print('Done!')

# Create a unique version of the dataset
df_unique = df[['Date', 'CP']].drop_duplicates().reset_index(drop = True)

# Shift prices: include only 
df_unique['lnCP1'] = np.log(df_unique['CP'].shift(-1))

df = pd.merge(
    df,               
    df_unique[['Date', 'lnCP1']],        
    on='Date',        
    how='left'        
)

df_new = pd.concat([df, pd.DataFrame(np.array(sentence_embeddings))], axis=1)

df_new.to_csv('EmbeddedData.csv', index=False)