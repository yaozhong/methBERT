# transformer implementaiton, improment module of BERT.

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
		std  = x.std(-1, keepdim=True)
		return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		# old implementation
		return x + self.dropout(sublayer(self.norm(x)))

		# on the testing
		#return self.norm(x + self.dropout(sublayer(x)))


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
def positional_norm_enc(att_center, len, att_std):
	position = torch.arange(0, len).float().unsqueeze(1).cuda()
	dist = torch.distributions.Normal(att_center, att_std)
	d = torch.exp_(dist.log_prob(position))
	return d

def positional_beta_enc(att_center, len, att_std):
	position = torch.arange(1, len+1).float().unsqueeze(1).cuda()
	position = position/(len+1)
	dist = torch.distributions.beta.Beta(att_center, att_std)
	d = torch.exp_(dist.log_prob(position))
	return d


# original implementaiton of positional embedding, used the same as in the BERT
class PositionalEmbedding_plus(nn.Module):
	def __init__(self, d_model, max_len=21):
		super().__init__()
		
		pe = torch.zeros(max_len, d_model).float()
		pe.require_grad = False
		
		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()

		pe[:, 0::2] = torch.sin(position * div_term) 
		pe[:, 1::2] = torch.cos(position * div_term) 

		pe = pe.unsqueeze(0)

		self.pe = torch.nn.parameter.Parameter(pe.float().cuda(), requires_grad=True)

	def forward(self, x):
		return self.pe[:, :x.size(1)]

#############################################################
# consider the relative distance in the attention module.
############################################################
class Relative_PositionalEmbedding(nn.Module):
	def __init__(self, num_units=21, max_relative_position=3):
		super().__init__()
		self.num_units = num_units
		self.max_relative_position = max_relative_position

		#self.embedding_table = torch.nn.parameter.Parameter(torch.Tensor(max_relative_position*2+1, num_units).cuda(), requires_grad=True) #.cuda()
		self.embedding_table = torch.nn.parameter.Parameter(torch.Tensor(max_relative_position*2+1, num_units), requires_grad=True).cuda()
		nn.init.xavier_uniform_(self.embedding_table)

	def forward(self, seq_len):

		range_vec_q = torch.arange(seq_len)
		range_vec_k = torch.arange(seq_len)

		dist_mat = range_vec_k[None,:] - range_vec_q[:,None]
		dist_mat_clipped = torch.clamp(dist_mat, -self.max_relative_position, self.max_relative_position)
		final_mat = dist_mat_clipped + self.max_relative_position
		final_mat = torch.LongTensor(final_mat)
		embeddings = self.embedding_table[final_mat]

		return embeddings


# used for paired ones
class SegmentEmbedding(nn.Embedding):
	def __init__(self, embed_size=32):
		super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
	def __init__(self, vocab_size=4, embed_size=32):
		super().__init__(vocab_size, embed_size, padding_idx=0)

# Ensembl of the embedding information
class BERTEmbedding_plus(nn.Module):
	def __init__(self, vocab_size, embed_size, dropout=0.1, inLineEmbed=True):
		super().__init__()

		#self.token   = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
		#self.segment  = SegmentEmbedding(embed_size=self.token.embedding_dim)

		self.position = PositionalEmbedding_plus(d_model=embed_size)
		self.dropout  = nn.Dropout(p=dropout)

		#self.inLineEmbed = inLineEmbed

		self.featEmbed = nn.Linear(vocab_size, embed_size)
		self.embed_size = embed_size

	def forward(self, sequence):
		x = self.featEmbed(sequence)  + self.position(sequence)   #+ self.segment(semgnet_label)
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

class Attention_relative(nn.Module):

	def forward(self, query, key, value, r_k, r_v, mask=None, dropout=None):
		scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(query.size(-1)) 
		
		# additional transform for the relative information
		batch_size, heads, length, _ =  query.size()
		r_q = query.permute(2,0,1,3).contiguous()
		r_q = r_q.reshape([length, heads*batch_size, -1])
		rel_score = torch.matmul(r_q, r_k.transpose(-2,-1))
		rel_score =  rel_score.contiguous().reshape([length, batch_size, heads, -1]).permute([1, 2, 0, 3])
		scores = scores + rel_score / math.sqrt(query.size(-1)) 
		

		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)

		p_attn = F.softmax(scores, dim=-1)

		if dropout is not None:
			p_attn = dropout(p_attn)

		# additional transform for the relative information
		
		r_attn = p_attn.permute(2,0,1,3).contiguous().reshape([length, heads*batch_size, -1])
		rel_v = torch.matmul(r_attn, r_v)
		rel_v = rel_v.contiguous().reshape([length, batch_size, heads, -1]).permute([1, 2, 0, 3])
		
		return torch.matmul(p_attn, value) + rel_v, p_attn


class MultiHeadedAttention_relative(nn.Module):
	def __init__(self, h, d_model, seq_len=21, dropout=0.1):
		super().__init__()

		assert d_model % h == 0
		self.d_k = d_model // h
		self.h = h

		self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
		self.output_linear = nn.Linear(d_model, d_model)
		self.attention = Attention_relative()
		self.dropout = nn.Dropout(p=dropout)

		SIDE_WINDOW_LIMIT=3
		# original, this will not correctly show the number of parameters
		self.r_v = Relative_PositionalEmbedding(self.d_k, SIDE_WINDOW_LIMIT)(seq_len)
		self.r_k = Relative_PositionalEmbedding(self.d_k, SIDE_WINDOW_LIMIT)(seq_len)
		
		self.attn_output = None
		

	def forward(self, query, key, value, mask=None):

		batch_size = query.size(0)

		query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) \
							for l, x in zip(self.linear_layers, (query, key, value))]

		x, attn = self.attention(query, key, value, self.r_k, self.r_v, mask=mask, dropout=self.dropout)
		x = x.transpose(1,2).contiguous().view(batch_size, -1, self.h * self.d_k)

		self.attn_output = attn

		return self.output_linear(x)


############ Transform block build ############ 

class TransformerBlock_relative(nn.Module):
	def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, seq_len):
		super().__init__()

		self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
		self.attention = MultiHeadedAttention_relative(h=attn_heads, d_model=hidden, seq_len=seq_len, dropout=0.1)
		self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout = dropout)
		self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, mask):
		x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
		x = self.output_sublayer(x, self.feed_forward)
		return self.dropout(x)

######################################
## BERT: bi-directional model build ##
######################################
class BERT_plus(nn.Module):
	def __init__(self, vocab_size=4, hidden=32, n_layers=3, attn_heads=1, dropout=0, motif_shift=0, motif_len=2, seq_len=21, inLineEmbed=True, device="cuda:0"):
		super().__init__()

		# note the network structure is defined in the inital function
		self.hidden = hidden
		self.n_layers = n_layers
		self.attn_heads = attn_heads

		# feed forward hidden
		self.feed_forward_hidden = hidden * 4

		# embedding module
		self.embedding = BERTEmbedding_plus(vocab_size, hidden, dropout, inLineEmbed)
		self.device = device

		# stacked transformers
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock_relative(hidden, attn_heads, hidden*4, dropout, seq_len) for _ in range(n_layers)])

		## key points to change hidden*5
		self.linear = nn.Linear(hidden*7, 2)
		self.tanh = nn.Tanh()

	def forward(self, x, segment_info=None):

		x = self.embedding(x)

		mask = None
		
		for transformer in self.transformer_blocks:
			x = transformer.forward(x, mask)

		mid_idx = int(x.size(1)/2)
		out = self.linear(torch.cat((x[:, mid_idx-3, :], x[:, mid_idx-2, :], x[:, mid_idx-1, :], x[:, mid_idx, :], 
			x[:, mid_idx+1, :], x[:, mid_idx+2, :], x[:, mid_idx+3, :]), -1))

		out = self.tanh(out)
		return out
		
