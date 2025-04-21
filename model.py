import torch

class QryTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
#
#
#

class DocTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

#
#
#



class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.ffw = torch.nn.Linear(in_features=embed_dim, out_features=vocab_size, bias=False)

    def forward(self, x):
        emb = self.emb(x)               # (batch_size, context_window, embed_dim)
        pooled = emb.mean(dim=1)        # (batch_size, embed_dim)
        out = self.ffw(pooled)          # (batch_size, vocab_size)
        return out

  

#
#
#
# if __name__ == '__main__':
#   model = CBOW(128, 8)
#   print('CBOW:', model)
#   criterion = torch.nn.CrossEntropyLoss()
#   inpt = torch.randint(0, 128, (3, 5)) # (batch_size, seq_len)
#   trgt = torch.randint(0, 128, (3,))   # (batch_size)
#   out = model(inpt)
#   loss = criterion(out, trgt)
#   print(loss) # ~ ln(1/128) --> 4.852...