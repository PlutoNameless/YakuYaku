import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

import copy

# 添加warmup scheduler
from torch.optim.lr_scheduler import LambdaLR

# 添加Beam Search
import torch.nn.functional as F

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=Constants.PAD)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe = self.pe[:,:x.size(1)] 
        x = x + pe
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % h == 0
        
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        
        # Do not apply dropout to input and output
        self.dropout = nn.Dropout(p=dropout)
        
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        
        self.attn = None
        
    def forward(self, query, key, value, mask=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,  
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = ResidualConnection(self_attn, dropout)
        self.feed_forward = ResidualConnection(feed_forward, dropout)

    def forward(self, x, mask):
        x = self.self_attn(x, x, x, mask)
        return self.feed_forward(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(layer.size, layer.size, bidirectional=True, batch_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size) 
        
    def forward(self, x, mask):
        x, _ = self.lstm(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = ResidualConnection(self_attn, dropout)
        self.src_attn = ResidualConnection(src_attn, dropout)
        self.feed_forward = ResidualConnection(feed_forward, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.src_attn(x, m, m, src_mask)
        return self.feed_forward(x)
        
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(layer.size, layer.size, batch_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        x, _ = self.lstm(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        return output
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    # 添加Beam Search
    def beam_search(self, src, src_mask, max_len, start_symbol, 
               beam_size=5, alpha=0.6, beta=0.6):

        memory = self.encode(src, src_mask)
        
        # 初始化beam
        candidates = [[start_symbol]] 
        log_probs = [0]
        lengths = [0]
        
        for step in range(max_len):
            new_candidates = []
            new_log_probs = []
            new_lengths = []
            
            # 对每个候选序列计算log prob
            for i, candidate in enumerate(candidates):
                last_token = candidate[-1]
                embedding = self.tgt_embed(last_token)
                output = self.decode(memory, src_mask, embedding, None)
                log_prob = F.log_softmax(self.generator(output[:, -1]), dim=-1)
                
                # Beam Search
                top_log_probs, indices = log_prob.topk(beam_size)
                for j in range(beam_size):
                    new_candidate = candidate + [indices[j].item()]
                    new_log_prob = log_probs[i] + top_log_probs[j]
                    new_length = lengths[i] + 1
                    
                    new_candidates.append(new_candidate)
                    new_log_probs.append(new_log_prob)
                    new_lengths.append(new_length)
                    
            # 添加长度惩罚
            for i, (new_log_prob, new_len) in enumerate(zip(new_log_probs, new_lengths)):
                new_log_prob /= (new_len + 1) ** beta
                new_log_probs[i] = new_log_prob
                
            # 按新分数排序    
            ordered = sorted(zip(new_candidates, new_log_probs), 
                            key=lambda x: x[1]/ (len(x[0])**alpha), 
                            reverse=True)
                            
            # 取top beam_size个
            candidates = [x[0] for x in ordered[:beam_size]]
            log_probs = [x[1] for x in ordered[:beam_size]]
            lengths = [len(x[0]) for x in ordered[:beam_size]]
        
        return candidates, log_probs
                
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):

    # 创建组件
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    # 构建Encoder和Decoder
    encoder = Encoder(
        EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(
        DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    
    # 其他模块定义
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position)) 
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    generator = Generator(d_model, tgt_vocab)
    
    # LSTM模块
    encoder.lstm = nn.LSTM(d_model, d_model, bidirectional=True, batch_first=True)
    decoder.lstm = nn.LSTM(d_model, d_model, batch_first=True)
    
    # 组装Transformer
    model = Transformer(encoder, decoder, src_embed, tgt_embed, generator)  
    
    # 参数初始化
    for name, p in model.named_parameters():
        if "lstm" in name:
             if "weight" in name:
                 nn.init.orthogonal_(p)
             elif "bias" in name:
                 nn.init.zeros_(p)
        else:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    return model

# 一个使用示例  
if __name__ == '__main__':
    vocab_size = 1000
    emb_size = 512
    N = 6
    epochs = 20
    
    model = make_model(vocab_size, vocab_size, N=N, d_model=emb_size)

    # 使用Adam优化器
    optimizer = Adam(model.parameters(), lr=0.0001)
    
    # 使用Cosine Annealing学习率衰减
    scheduler = CosineAnnealingLR(optimizer, epochs)
    
    # 使用交叉熵损失
    loss_fn = nn.NLLLoss() 

    train_loader = DataLoader(train_data, batch_size=64)

    device = torch.device('cuda')
    model.to(device)

    # 使用混合精度训练
    scaler = GradScaler()
    
    # 学习率warmup
    warmup_steps = 4000
    warmup_scheduler = LambdaLR(optimizer, linear_warmup_decay(warmup_steps, epochs*len(train_loader)))

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with autocast():
                output = model(x, y, None, None)
                loss = loss_fn(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新warmup学习率
            warmup_scheduler.step()

            if i%100 == 0:
                print(f'Epoch {epoch} iteration {i}: loss {loss.item()}')

        # 更新cosine annealing学习率
        scheduler.step()
    
    # Beam Search预测
    src = ... # 源语言输入
    src_mask = ... # 源语言 Attention Mask
    candidates, log_probs = model.beam_search(src, src_mask, max_len=20, start_symbol=1, beam_size=5)

    # 保存和加载
    torch.save(model.state_dict(), 'model.pth')
    model = Model()
    model.load_state_dict(torch.load('model.pth'))
