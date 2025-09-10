import torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


epochs = 10000
world_size = 4
save_every = 1000

embd_size = 384
batch_size = 64
context_size = 256
dropout_thres = 0.2
n_heads = 6
n_layers = 6
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


# LOAD DATA
with open("data/input.txt", "r") as f:
  text = f.read()

vocab_list = ''.join(sorted(list(set(text))))
vocab_size = len(vocab_list)

ctoi = { val: i for i, val in enumerate(vocab_list)}
itoc = { i: val for i, val in enumerate(vocab_list)}
encoder = lambda s: [ctoi[i] for i in s]
decoder = lambda enc: ''.join([itoc[i] for i in enc])

enc = encoder(text)
tt_split = 0.9

test = torch.tensor(enc[int(tt_split*len(enc)):])
train = torch.tensor(enc[:int(tt_split*len(enc))])


# CLASSES
class SelfAttentionHead(torch.nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.head_size = head_size
    self.key = torch.nn.Linear(embd_size, head_size, bias=False)
    self.value = torch.nn.Linear(embd_size, head_size, bias=False)
    self.query = torch.nn.Linear(embd_size, head_size, bias=False)

    self.drop = torch.nn.Dropout(dropout_thres)
    self.register_buffer("tril", torch.tril(torch.ones((context_size, context_size))))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, head_size)
    v = self.value(x) # (B, T, head_size)
    q = self.query(x) # (B, T, head_size)

    W = q @ k.transpose(-1, -2) * self.head_size**(-0.5) # (B, T, T)
    W.masked_fill_(self.tril[:T, :T] == 0, float("-inf"))
    w_mask = torch.nn.functional.softmax(W, dim=-1) # (B, T, T)
    w_mask = self.drop(w_mask)

    return w_mask @ v # (B, T, head_size)

class MultiAttentionHead(torch.nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = torch.nn.ModuleList(
        SelfAttentionHead(head_size) for _ in range(n_heads)
    )
    self.proj = torch.nn.Linear(n_heads * head_size, embd_size)
    self.drop = torch.nn.Dropout(dropout_thres)

  def forward(self, x):
    st = torch.cat([head(x) for head in self.heads], dim=-1)
    return self.drop(self.proj(st))

class FeedForwardLayer(torch.nn.Module):
  def __init__(self, nin, nout):
    super().__init__()
    self.layers = torch.nn.Sequential(
        torch.nn.Linear(nin, 4*nout),
        torch.nn.ReLU(),
        torch.nn.Linear(4*nout, nout),
        torch.nn.Dropout(dropout_thres)
    )

  def forward(self, x):
    return self.layers(x)

class Block(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.sa_heads = MultiAttentionHead(n_heads, embd_size // n_heads)
    self.ffwd = FeedForwardLayer(embd_size, embd_size)

    self.ln1 = torch.nn.LayerNorm((embd_size,))
    self.ln2 = torch.nn.LayerNorm((embd_size,))

  def forward(self, x):
    '''
    in:
    - x: tensor (batch_size, context_size, embd_size)
    out:
    - out: tensor (batch_size, context_size, embd_size)
    '''
    x = self.sa_heads(self.ln1(x)) + x # residual connections
    x = self.ffwd(self.ln2(x)) + x
    return x

class BigramLanguageModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.feat_encoding = torch.nn.Embedding(vocab_size, embd_size)
    self.pos_encoding = torch.nn.Embedding(context_size, embd_size)

    self.blocks = torch.nn.Sequential(
        *[Block() for _ in range(n_layers)],
        torch.nn.LayerNorm((embd_size,)),
        torch.nn.Linear(embd_size, vocab_size)
    )

  def forward(self, xs, ys=None): # outputs logits, loss
    '''
    in:
    - xs: tensor (batch_size, context_size)
    - ys: tensor (batch_size, context_size) or None
    out:
    - logits: tensor (batch_size, context_size, vocab_size)
    - loss: tensor (1,) or None
    '''

    f = self.feat_encoding(xs) # (batch_size, context_size, embd_size)
    p = self.pos_encoding(torch.arange(0, xs.shape[1], device=device)) # (context_size, embd_size)

    x = f + p
    logits = self.blocks(x)
    B, C, V = logits.shape

    if ys is not None:
      logits_ce = logits.reshape(B*C, V)
      ys_ce = ys.reshape(B*C)

      loss = torch.nn.functional.cross_entropy(logits_ce, ys_ce)
    else:
      loss = None
    return logits, loss

  def generator(self, max_length, batch_size_):
    '''
    in:
    - max_length: int
    out:
    - out: tensor (batch_size, max_length)
    '''
    out = torch.zeros((batch_size_, 1), dtype=torch.long, device=device)

    for _ in range(max_length - 1):
      last_char = out[:, -context_size:]

      logits, _ = self.forward(last_char)

      logits = logits[:, -1, :]
      probs = torch.nn.functional.softmax(logits, dim=-1)
      ntoken = torch.multinomial(probs, 1)
      out = torch.cat((out, ntoken), dim=1)

    return out

class Trainer:
    def __init__(self, model, train_data, optimizer, gpu_id, save_every):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.save_every = save_every


    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        x, y = next(iter(self.train_data))
        x, y = x.to(self.gpu_id), y.to(self.gpu_id)
        logits, loss = self.model(x, y)

        self.optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            self.optimizer.step()

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def loader_setup():
    starts = torch.arange(0, len(train) - context_size - 1)
    sampler = DistributedSampler(starts, shuffle=True)
    def collate(starts_batch):
        xb = torch.stack([train[i:i+context_size] for i in starts_batch])
        yb = torch.stack([train[i+1:i+1+context_size] for i in starts_batch])
        return xb, yb

    loader = DataLoader(starts, batch_size=batch_size, sampler=sampler, collate_fn=collate, drop_last=True)
    return loader


def main(rank, world_size):
    ddp_setup(rank, world_size)
    model = BigramLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = loader_setup()

    t = Trainer(model, loader, optimizer, rank, save_every)
    t.train(epochs)

    destroy_process_group()

if __name__ == "__main__":
    mp.spawn(main, args=(world_size,), nprocs=world_size)


