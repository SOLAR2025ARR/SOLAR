import torch
from torch import nn
from torch.nn import functional as F

class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, dropout_rate: float):
        super(PointWiseFeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = relu(xW1 + b1)W2 + b2
        # x: (batch_size, seq_len, hidden_size)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class SASRec(nn.Module):

    def __init__(self, num_users: int, num_items: int, 
                 maxlen: int, hidden_size: int, dropout_rate: float, 
                 num_blocks: int, num_heads: int, device: str):
        super(SASRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.device = torch.device(device)

        self.item_embedding = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=0)
        self.positional_embedding = nn.Embedding(self.maxlen + 1, self.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_rate)

        self.attention_layernorms = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=1e-6) for _ in range(self.num_blocks)])
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(self.hidden_size, self.num_heads, self.dropout_rate) for _ in range(self.num_blocks)])
        self.ffn_layernorms = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=1e-6) for _ in range(self.num_blocks)])
        self.ffn_layers = nn.ModuleList([PointWiseFeedForward(self.hidden_size, self.dropout_rate) for _ in range(self.num_blocks)])

        self.last_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def forward_feats(self, seqs):
        bs = len(seqs)
        seqlen = len(seqs[0])
        # seqs: (batch_size, seq_len)
        seqs = torch.tensor(seqs, dtype=torch.long, device=self.device) # (batch_size, seq_len)
        item_embs = self.item_embedding(seqs) # (batch_size, seq_len, hidden_size)
        item_embs *= self.hidden_size ** 0.5
        # position mask
        positions = torch.arange(1, seqlen + 1).repeat(bs, 1).to(self.device) # (batch_size, seq_len) 

        positions.masked_fill_(seqs == 0, 0)
        pos_embs = self.positional_embedding(positions) # (batch_size, seq_len, hidden_size)

        x = self.emb_dropout(item_embs + pos_embs)

        # calsual attention mask
        causal_mask = 1 - torch.triu(torch.ones((seqlen, seqlen)), diagonal=1).to(self.device) # (seq_len, seq_len)

        for i in range(self.num_blocks):
            # self attention
            # pytorch's multiheadattention forward function takes inputs in batch second format
            x = x.permute(1, 0, 2) # (seq_len, batch_size, hidden_size)
            old_x = x
            x = self.attention_layernorms[i](x)
            x, _ = self.attention_layers[i](x, x, x, attn_mask=causal_mask)
            x = x + old_x
            x = x.permute(1, 0, 2)
            x = x + self.ffn_layers[i](self.ffn_layernorms[i](x))
        x = self.last_layernorm(x) # (batch_size, seq_len, hidden_size)
        return x

    def forward(self, user_ids, seqs, pos_seqs, neg_seqs): # for training
        """
        Args:
            user_ids: list of user ids
            seqs: list of user historical sequences
            pos_seqs: list of positive item sequences
            neg_seqs: list of negative item sequences
        Returns:
            pos_logits: the matching score between user historical sequences and pos_seqs
            neg_logits: the matching score between user historical sequences and neg_seqs
        """
        seq_feats = self.forward_feats(seqs) # (batch_size, seq_len, hidden_size)

        pos_embs = self.item_embedding(torch.LongTensor(pos_seqs).to(self.device)) # (batch_size, seq_len, hidden_size)
        neg_embs = self.item_embedding(torch.LongTensor(neg_seqs).to(self.device)) # (batch_size, seq_len, hidden_size)

        # the dot product of feat and item embedding is the matching score
        pos_logits = torch.sum(seq_feats * pos_embs, dim=-1) # (batch_size, seq_len)
        neg_logits = torch.sum(seq_feats * neg_embs, dim=-1) # (batch_size, seq_len)

        return pos_logits, neg_logits

    def predict(self, user_ids, seqs, item_ids):
        """
        Args:
            user_ids: list of user ids
            seqs: list of user historical sequences
            item_ids: list of item ids to be predicted
        Returns:
            logits: the matching score between user historical sequences and item_ids
        """
        seq_feats = self.forward_feats(seqs) # (batch_size, seq_len, hidden_size)
        final_feat = seq_feats[:, -1, :] # (batch_size, hidden_size)
        item_embs = self.item_embedding(torch.LongTensor(item_ids).to(self.device)) # (num_items, hidden_size)

        # the dot product of final_feat and item embedding is the matching score
        logits = torch.matmul(final_feat, item_embs.t()) # (batch_size, num_items)

        return logits # (batch_size, num_items)