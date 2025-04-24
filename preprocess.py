import re
import torch

def text_to_embedding(text, word_to_id, embedding_matrix, unk_id=0):
    '''
    get the embeddings of the tokens of the text from the embedding matrix
     1. tokenize the text into tokens
     2. look up the tokens' embeddings from the embedding matrix
           if any token is not in the embedding matrix, use the unk_id
     3. return the embeddings of tokens

     embedding_matrix: key=word id, value=tensor of embedding
    '''
    # remove punctuation and non alphabetic characters
    remove_punctuation = re.sub(r'[^\w\s]', '', text)
    lower_case_words = remove_punctuation.lower()
    tokens = lower_case_words.split()#split by whitespace, tab and newline
    ids = [word_to_id.get(tok, unk_id) for tok in tokens]
    embeddings = torch.stack([
        embedding_matrix[i] if i < embedding_matrix.size(0) else embedding_matrix[unk_id]
        for i in ids
    ])
    return embeddings
