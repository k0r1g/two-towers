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
  def __init__(self, voc, emb):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)

  def forward(self, inpt):
    emb = self.emb(inpt)
    emb = emb.mean(dim=1)
    out = self.ffw(emb)
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