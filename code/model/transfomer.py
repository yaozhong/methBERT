# basic transformer implementaiton

import torch, math
import torch.nn as nn
import torch.nn.functional as F

### basic modules ###
class GELU(nn.Module):
	def forward(self, x):
		return 0.5 * x * (1+torch.tanh(math.sqrt(2 / math.pi) *(x + 0.044715 * torch.pow(x, 3)))) 

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = GELU()

	def forward(self, x):
		return self.w_2(self.dropout(self.activation(self.w_1(x))))

###################################
#  Embedding 
###################################
class PositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_len=32):
		super().__init__()
		pe = torch.zeros(max_len, d_model).float()
		pe.require_grad = False

		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return self.pe[:, :x.size(1)]

# used for paired ones
class SegmentEmbedding(nn.Embedding):
	def __init__(self, embed_size=32):
		super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
	def __init__(self, vocab_size=4, embed_size=256):
		super().__init__(vocab_size, embed_size, padding_idx=0)

# Ensembl of the embedding information
class BERTEmbedding(nn.Module):
	def __init__(self, vocab_size, embed_size, dropout=0):
		super().__init__()
		self.token 	  = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
		self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
		self.segment  = SegmentEmbedding(embed_size=self.token.embedding_dim)
		self.dropout  = nn.Dropout(p=dropout)
		self.featEmbed = nn.Linear(7, embed_size)
		self.embed_size = embed_size

	def forward(self, sequence):
		# testing models without positional embedding
		x = self.featEmbed(sequence) # + self.position(sequence) #+ self.segment(semgnet_label)
		#x = self.token(sequence) + self.position(sequence) #+ self.segment(semgnet_label)
		return self.dropout(x)
		

############ attention ############ 
class Attention(nn.Module):

	def forward(self, query, key, value, mask=None, dropout=None):
		scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(query.size(-1))
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)

		p_attn = F.softmax(scores, dim=-1)
		if dropout is not None:
			p_attn = dropout(p_attn)

		return torch.matmul(p_attn,value), p_attn

class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super().__init__()
		assert d_model % h == 0

		self.d_k = d_model // h
		self.h = h

		self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
		self.output_linear = nn.Linear(d_model, d_model)
		self.attention = Attention()
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		batch_size = query.size(0)

		query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) \
							for l, x in zip(self.linear_layers, (query, key, value))]

		x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
		x = x.transpose(1,2).contiguous().view(batch_size, -1, self.h * self.d_k)

		return self.output_linear(x)

############ Transform block build ############ 

class TransformerBlock(nn.Module):
	def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
		super().__init__()
		self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
		self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout = dropout)
		self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
		self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, mask):
		x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
		x = self.output_sublayer(x, self.feed_forward)
		return self.dropout(x)

######################################
## BERT: bi-directional model build ##
######################################
class BERT(nn.Module):
	def __init__(self, vocab_size=4, hidden=32, n_layers=3, attn_heads=8, dropout=0):
		super().__init__()

		self.hidden = hidden
		self.n_layers = n_layers
		self.attn_heads = attn_heads
		self.feed_forward_hidden = hidden * 4

		self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

		# stacked transformers
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock(hidden, attn_heads, hidden*4, dropout) for _ in range(n_layers)])

		self.linear = nn.Linear(hidden, 2)


	def forward(self, x, segment_info=None):
		#mask = (x>0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
		x = self.embedding(x)
		mask = None

		for transformer in self.transformer_blocks:
			x = transformer.forward(x, mask)

		# add additional forward calcuation, this part to be confirmed
		x = self.linear(x[:, 0])

		return x

