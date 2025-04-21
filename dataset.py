#download huggingface datasets
#install the datasets library
#pip install datasets
#load the datasets library
#download ms_marco dataset

from datasets import load_dataset
import tokenizer as tk

# Load the default config and split (defaults to split='train' if available)
dataset = load_dataset("microsoft/ms_marco", 'v1.1')

# See what's inside
#print(dataset)

#get the training data
train_data = dataset['train']
#print(train_data)

#print(train_data[0])

#get the test data
test_data = dataset['validation']

#get the validation data
validation_data = dataset['test']
#iterate through the whole training data
import numpy as np

query2passage = {}
total_pos = 0
total_neg = 0 
for example in train_data:
    query = example['query']
    passages = example['passages']
    mask=passages['is_selected'] #[True, False, True, False, True]
    no_of_pos=sum(mask)
    no_of_neg=len(mask)-no_of_pos
    total_pos+=no_of_pos
    total_neg+=no_of_neg
    ps = np.array(passages['passage_text']) #['apple', 'banana', 'cherry', 'date', 'elderberry']
    mask = np.array(mask, dtype=bool) # [True, False, True, False, True]
    ps = np.ravel(ps)
    mask = np.ravel(mask)
    pos_passages = ps[mask]
    neg_passages = ps[~mask]
    query2passage[query] = (pos_passages, neg_passages)
print("Total number of positive passages: ", total_pos)
print("Total number of negative passages: ", total_neg)

import torch

torch.save(query2passage, 'query2passage.pt')
print("Query to passage mapping saved")
torch.save(pos_passages, 'pos_passages.pt')
print("Positive passages saved")
torch.save(neg_passages, 'neg_passages.pt')
print("Negative passages saved")
corpus_list = []
for example in train_data:
    passages = example['passages']
    ps = np.array(passages['passage_text']) #['apple', 'banana', 'cherry', 'date', 'elderberry']
    corpus_list.extend(ps)
print("finished creating corpus_list")
corpus = ''.join(corpus_list)
print(corpus[:1000])
#tokenise the corpus
word_to_id, id_to_word, filtered_corpus=tk.tokenizer(corpus, top_k=30000)
torch.save(filtered_corpus, 'filtered_corpus.pt')
print("Filtered corpus saved")
vocab={word for word in word_to_id.keys()}
print("Vocabulary size: ", len(vocab))