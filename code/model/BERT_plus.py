# transformer implementaiton, improment module of BERT.

import torch, math
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN=21
SIDE_WINDOW_LIMIT=3

PRIOR_STD=5.0
CENTER_POS=10.0

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

# absolute position encoding.  
class PositionalEmbedding_weight(nn.Module):
	def __init__(self, d_model,  motif_shift, motif_len, max_len=SEQ_LEN):
		super(PositionalEmbedding_weight, self).__init__()

		self.pe = torch.zeros(max_len, d_model).float().cuda()
		self.pe.require_grad = False

		# 6ma, M_dam_gAtc
		self.weight = torch.tensor([ 0.0129,  0.0140,  0.0111,  0.0137,  0.0128,  0.0100,  0.0070,  0.0247,0.0193, -0.0162, 
		 0.0748,  0.0838,  0.0118,  0.0029,  0.0077,  0.0098, 0.0100,  0.0120,  0.0104,  0.0106,  0.0120])
		self.weight_var = torch.tensor([0.2221, 0.2293, 0.2312, 0.2214, 0.2240, 0.2301, 0.2206, 0.2287, 0.2540,
			0.2597, 0.2845, 0.4460, 0.2490, 0.2177, 0.2264, 0.2227, 0.2221, 0.2280, 0.2223, 0.2218, 0.2309])

		self.weight = torch.abs(self.weight)
		self.weight_norm = self.weight / torch.sum(self.weight)
		self.weight = self.weight_norm.unsqueeze(1).cuda()
		#self.weight = self.weight.unsqueeze(1).cuda()

		position = torch.arange(0, max_len).float().unsqueeze(1).cuda()
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp().cuda()

		self.pe[:, 0::2] = torch.sin(position * div_term) 
		self.pe[:, 1::2] = torch.cos(position * div_term) 

		self.pe = self.pe.unsqueeze(0)  # expand one dimention.	

		self.pe = self.pe * self.weight * self.weight_var.unsqueeze(1).cuda()

		# paramterize
		self.pe = torch.nn.parameter.Parameter(self.pe.float().cuda(), requires_grad=True)

	def forward(self, x):
		return self.pe[:, :x.size(1)]

# original implementaiton of positional embedding, used the same as in the BERT

class PositionalEmbedding_plus(nn.Module):
	def __init__(self, d_model, max_len=SEQ_LEN):
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
	def __init__(self, num_units=SEQ_LEN, max_relative_position=SIDE_WINDOW_LIMIT):
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


class BERTEmbedding_weight(nn.Module):
	def __init__(self, vocab_size, embed_size, dropout=0.1, inLineEmbed=True, motif_shift=0, motif_len=2):
		super().__init__()

		self.position_w = PositionalEmbedding_weight(d_model=embed_size, motif_shift=motif_shift, motif_len=motif_len)
		self.dropout  = nn.Dropout(p=dropout)

		self.featEmbed = nn.Linear(vocab_size, embed_size)
		self.embed_size = embed_size
		self.vocab_size = vocab_size

	def forward(self, sequence):

		x = self.featEmbed(sequence)  + self.position_w(sequence)
		return self.dropout(x)

# adding the weight to the feature level
class BERTEmbedding_weight2_noused(nn.Module):
	def __init__(self, vocab_size, embed_size, dropout=0.1, inLineEmbed=True, motif_shift=0, motif_len=2):
		super().__init__()

		self.att_center = nn.Parameter(torch.Tensor([int(SEQ_LEN/2)+motif_shift]).cuda(), requires_grad=True)
		self.att_std    = nn.Parameter(torch.Tensor([int(motif_len/2)+1]).cuda(), requires_grad=True)
		self.weight = positional_norm_enc(self.att_center, SEQ_LEN, self.att_std)

		self.token 	  = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
		self.position_w = PositionalEmbedding(d_model=self.token.embedding_dim)
		self.segment  = SegmentEmbedding(embed_size=self.token.embedding_dim)
		self.dropout  = nn.Dropout(p=dropout)

		self.inLineEmbed = inLineEmbed
		# replace the feature Embeding to be the linear
		self.featEmbed = nn.Linear(vocab_size, embed_size)
		self.embed_size = embed_size

	def forward(self, sequence):
		# testing models without positional embedding
		if self.inLineEmbed:
			x = self.featEmbed(sequence*self.weight)  + self.position_w(sequence)   #+ self.segment(semgnet_label)		

		else:
			x = self.token(sequence) # + self.position(sequence)     #+ self.segment(semgnet_label)

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
	def __init__(self, h, d_model, dropout=0.1):
		super().__init__()

		assert d_model % h == 0
		self.d_k = d_model // h
		self.h = h

		self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
		self.output_linear = nn.Linear(d_model, d_model)
		self.attention = Attention_relative()
		self.dropout = nn.Dropout(p=dropout)

		# original, this will not correctly show the number of parameters
		self.r_v = Relative_PositionalEmbedding(self.d_k, SIDE_WINDOW_LIMIT)(SEQ_LEN)
		self.r_k = Relative_PositionalEmbedding(self.d_k, SIDE_WINDOW_LIMIT)(SEQ_LEN)
		
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
	def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
		super().__init__()

		self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
		self.attention = MultiHeadedAttention_relative(h=attn_heads, d_model=hidden)
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
	def __init__(self, vocab_size=4, hidden=32, n_layers=3, attn_heads=1, dropout=0, motif_shift=0, motif_len=2, inLineEmbed=True, device="cuda:0"):
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
			[TransformerBlock_relative(hidden, attn_heads, hidden*4, dropout) for _ in range(n_layers)])

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
		

# 20201225 updated new position
class BERT_position(nn.Module):
	def __init__(self, vocab_size=4, hidden=32, n_layers=3, attn_heads=1, dropout=0, motif_shift=0, motif_len=2, inLineEmbed=True, device="cuda:0"):
		super().__init__()

		self.hidden = hidden
		self.n_layers = n_layers
		self.attn_heads = attn_heads

		# feed forward hidden
		self.feed_forward_hidden = hidden * 4

		# embedding module
		self.embedding = BERTEmbedding_weight(vocab_size, hidden, dropout, inLineEmbed)
		self.device = device

		# stacked transformers
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock_relative(hidden, attn_heads, hidden*4, dropout) for _ in range(n_layers)])

		self.linear = nn.Linear(hidden, 2)

	def forward(self, x, segment_info=None):

		x = self.embedding(x)
		mask = None
		
		for transformer in self.transformer_blocks:
			x = transformer.forward(x, mask)

		out = self.linear(x[:, int(x.size(1)/2), :])

		return out.to(self.device)

############### followng modules are under-testing #################
## not used in the current stage
class BERT_plus_rnn(nn.Module):
	def __init__(self, vocab_size=4, hidden=32, n_layers=3, attn_heads=1, dropout=0, inLineEmbed=True, device="cuda:0"):
		super().__init__()

		self.hidden = hidden
		self.n_layers = n_layers
		self.attn_heads = attn_heads
		self.feed_forward_hidden = hidden * 4
		self.embedding = BERTEmbedding_weight(vocab_size, hidden, dropout, inLineEmbed)
		self.device = device

		# stacked transformers
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock_relative(hidden, attn_heads, hidden*4, dropout) for _ in range(n_layers)])

		self.linear = nn.Linear(hidden*2, 2)
		self.inLineEmbed = inLineEmbed

		self.fc0 = nn.Linear(hidden, 32)
		self.fc1 = nn.Linear(32, 2)

		self.fc_midRound = nn.Linear(hidden*5, 2)

		self.lstm = nn.LSTM(hidden, hidden, 1, batch_first=True, bidirectional=True)

	def forward(self, x, segment_info=None):

		# previous embedding approach
		x = self.embedding(x)
		mask = None

		for transformer in self.transformer_blocks:
			x = transformer.forward(x, mask)

		h0 = torch.zeros(1*2, x.size(0), self.hidden).to(self.device)
		c0 = torch.zeros(1*2, x.size(0), self.hidden).to(self.device)

		x, _ = self.lstm(x, (h0,c0))
	
		#2. direct output without relu
		out = self.linear(x[:, int(x.size(1)/2), :])

		return out


class BERT_plus_rnn2(nn.Module):
	def __init__(self, vocab_size=4, hidden=32, n_layers=3, attn_heads=1, dropout=0, inLineEmbed=True, device="cuda:0"):
		super().__init__()

		self.hidden = hidden
		self.n_layers = n_layers
		self.attn_heads = attn_heads
		self.feed_forward_hidden = hidden * 4
		self.embedding = BERTEmbedding(vocab_size, hidden, dropout, inLineEmbed)
		self.device = device

		# stacked transformers
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock_relative(hidden, attn_heads, hidden*4, dropout) for _ in range(n_layers)])

		self.linear = nn.Linear(hidden*2, 2)
		self.inLineEmbed = inLineEmbed

		self.fc0 = nn.Linear(hidden, 32)
		self.fc1 = nn.Linear(32, 2)

		self.fc_midRound = nn.Linear(hidden*5, 2)

		self.lstm1 = nn.LSTM(vocab_size, int(hidden/2), 1, batch_first=True, bidirectional=True)
		self.lstm2 = nn.LSTM(hidden,     hidden, 1, batch_first=True, bidirectional=True)

	def forward(self, x, segment_info=None):

		mask = None
	
		h00 = torch.zeros(1*2, x.size(0), int(self.hidden/2)).to(self.device)
		c00 = torch.zeros(1*2, x.size(0), int(self.hidden/2)).to(self.device)
		x, _ = self.lstm1(x, (h00,c00))

		# using the rnn-embedding approach
		for transformer in self.transformer_blocks:
			x = transformer.forward(x, mask)

		h01 = torch.zeros(1*2, x.size(0), self.hidden).to(self.device)
		c01 = torch.zeros(1*2, x.size(0), self.hidden).to(self.device)

		x, _ = self.lstm2(x, (h01,c01))
		out = self.linear(x[:, int(x.size(1)/2), :])

	
		return out
