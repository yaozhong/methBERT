""" multiple processing and caching extracted features """

from dataProcess.ref_util import get_fast5s, sim_seq_singal, pdist
from dataProcess.feature_extract import single_read_process
import multiprocessing as mp
import numpy as np
import time, tqdm
import random
from itertools import combinations

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

# according to the data, the final real number may be var.
nc_dic ={'A':0, 'T':1, 'G':2, 'C':3, 'N':4}

# to testing , not used any more (1). too large for loading into the memory, (2). not owrk for more parameters setting.
def extract_sample_fold(fast5_dir, label=0):

	file_list = get_fast5s(fast5_dir)
	print("[@] Start extracting input data from fast files:")
	print("  * [%d] fast5-files in [%s]" %(len(file_list), fast5_dir))
	cores = mp.cpu_count()
	print("* Using [%d] CPU-cores" %(cores))

	#parallel allocation of the extraction
	pool = mp.Pool(processes=cores)

	output = list(tqdm.tqdm(pool.imap(single_read_process, file_list), total=len(file_list)))
	#output = pool.map(single_read_process, file_list)

	pool.close()
	pool.join()

	print("Total number files processed: %d" %len(output))
	print("Effective prcessed number: %d" %(len([o for o in output if o is not None])))


# 20200503 pytorch data load
class MethylDataSet(Dataset):
	def __init__(self, fast5_files, label, motif, m_shift, w_len, transform=None):
		self.fast5_files = fast5_files
		self.label = label
		self.transform = transform
		self.motif = motif
		self.m_shift = m_shift
		self.w_len = w_len

	def __getitem__(self,index):
		# process the target read
		feat_list = single_read_process(self.fast5_files[index], self.motif, self.m_shift, self.w_len)

		# any transormation
		if self.transform:
			feat_list = self.transform(feat_list)

		if feat_list is not None:
			return feat_list, [self.label]*len(feat_list)
		
	def __len__(self):
		return len(self.fast5_files)

class pairedSampleSet(Dataset):
	def __init__(self, X1, X2, pLabel, device):
		self.X1 = X1.to(device)
		self.X2 = X2.to(device)
		self.pLabel = pLabel

	def __getitem__(self, index):
		return self.X1[index], self.X2[index], self.pLabel[index]

	def __len__(self):
		return len(self.pLabel)


# 2020/08/03 load by sequence
## in use loading 
class seqTripletSampleSet(Dataset):
	def __init__(self, seq_group_data, device):
		self.seq_group_data = seq_group_data
		self.seqs = list(seq_group_data.keys())
		self.data_size = len(self.seqs)
		self.num_limit = 5

	def __getitem__(self, index):

		triplets = []
		s = self.seqs[index]

		num_meth   = len(self.seq_group_data[s][1])
		num_unMeth = len(self.seq_group_data[s][0])

		if num_meth == 0 or num_unMeth == 0:
			return None

		# needs index tensor
		seq = F.one_hot(torch.LongTensor([int(x) for x in s]))

		# positive smaple
		if num_meth > 1:
			if num_meth > self.num_limit:
				a_p_idx = random.sample(range(num_meth), self.num_limit)
			else:
				a_p_idx = range(num_meth)

			a_p_idx = list(combinations(a_p_idx, 2))
			
			anchors   = [ torch.cat((seq.float(), self.seq_group_data[s][1][idx[0]]), -1) for idx in a_p_idx ]
			positives = [ torch.cat((seq.float(), self.seq_group_data[s][1][idx[1]]), -1) for idx in a_p_idx ]

			"""
			anchors   = [ self.seq_group_data[s][1][idx[0]] for idx in a_p_idx ]
			positives = [ self.seq_group_data[s][1][idx[1]] for idx in a_p_idx ]
			"""

			# negative samples
			if num_unMeth > self.num_limit:
				n_idx = random.sample(range(num_unMeth), self.num_limit)
			else:
				n_idx = range(num_unMeth)

			negatives = [ torch.cat((seq.float(), self.seq_group_data[s][0][idx]), -1) for idx in n_idx ]
			# negatives = [ self.seq_group_data[s][0][idx] for idx in n_idx ]

			# group triplet data
			for i in range(len(a_p_idx)):
				for j in range(len(n_idx)):
					temp_triplets = torch.stack([anchors[i], positives[i], negatives[j]], dim=0)
					triplets.append(temp_triplets)

		# why zero cases
		if len(triplets) > 0:
			triplets = torch.stack(triplets, dim=0) #.permute(1,0,2,3) 
			return triplets
		else:
			return None

	def __len__(self):
		return self.data_size


# 2020/08/09 Triplet generation
class seqTripletSampleSet_withHardness(Dataset):
	def __init__(self, seq_group_data, device, net):
		self.seq_group_data = seq_group_data
		self.seqs = list(seq_group_data.keys())
		self.data_size = len(self.seqs)
		self.num_limit = 5
		self.net = net
		self.device = device

	def __getitem__(self, index):

		triplets = []
		s = self.seqs[index]

		num_meth   = len(self.seq_group_data[s][1])
		num_unMeth = len(self.seq_group_data[s][0])

		if num_meth == 0 or num_unMeth == 0:
			return None

		# needs index tensor
		seq = F.one_hot(torch.LongTensor([int(x) for x in s]))

		# positive smaple
		if num_meth > 1 and num_unMeth >= 1:
			if num_meth > self.num_limit:
				a_p_idx = random.sample(range(num_meth), self.num_limit)
			else:
				a_p_idx = range(num_meth)

			a_p_idx = list(combinations(a_p_idx, 2))
			anchors   = [ torch.cat((seq.float(), self.seq_group_data[s][1][idx[0]]), -1) for idx in a_p_idx ]
			positives = [ torch.cat((seq.float(), self.seq_group_data[s][1][idx[1]]), -1) for idx in a_p_idx ]

			# can add permuation here
			n_idx = list(range(num_unMeth))
			random.shuffle(n_idx)
			negatives = [ torch.cat((seq.float(), self.seq_group_data[s][0][idx]), -1) for idx in n_idx ]
			
			self.net.eval()
			with torch.no_grad():
				anchors_embed   = self.net.get_embedding(torch.stack(anchors, dim=0).to(self.device)).cpu()
				positives_embed = self.net.get_embedding(torch.stack(positives, dim=0).to(self.device)).cpu()
				negatives_embed = self.net.get_embedding(torch.stack(negatives, dim=0).to(self.device)).cpu()

			#print(anchors_embed.shape, positives_embed.shape, negatives_embed.shape)
			a_p = torch.nn.PairwiseDistance(p=2)(anchors_embed, positives_embed) 
			a_n = pdist(anchors_embed, negatives_embed)

			#print(a_p.shape, a_n.shape)

			# hard selection
			for i in range(len(a_p)):
				add_num = 0
				for j in range(len(negatives)):
					# hard example, not suggested to use why
					"""
					if(a_n[i,j] < a_p[i]):
						temp_triplets = torch.stack([anchors[i], positives[i], negatives[j]], dim=0)
						triplets.append(temp_triplets)
						add_num += 1
						
						if add_num > self.num_limit:
							break
			
					# if no hard example add semi-hard
					"""
					if(a_n[i,j] < a_p[i]+1 and a_n[i,j] > a_p[i]):
						temp_triplets = torch.stack([anchors[i], positives[i], negatives[j]], dim=0)
						triplets.append(temp_triplets)
						add_num += 1

						if add_num > self.num_limit:
							break
					
				

		# why zero cases
		if len(triplets) > 0:
			triplets = torch.stack(triplets, dim=0) #.permute(1,0,2,3) 
			return triplets
		else:
			return None

	def __len__(self):
		return self.data_size


def my_collate2(batch):
	batch = list(filter(lambda x : x is not None, batch))
	if len(batch) > 0:
		batch = torch.cat(batch,dim=0)
		triplets = batch.permute(1,0,2,3)
		return triplets
	else:
		return None

# all-into the memory loading
class tripletSampleSet(Dataset):
	def __init__(self, X1, X2, X3, device):
		self.X1 = X1.to(device)
		self.X2 = X2.to(device)
		self.X3 = X3.to(device)

	def __getitem__(self, index):
		return self.X1[index], self.X2[index], self.X3[index]

	def __len__(self):
		return len(self.X1)


################################################################
## process and filtering the loading reads
## expand the read-level sample to each motif level sample
################################################################
def collate_fn(batch):
	"""
	batch = list(filter(lambda x : x is not None, batch))
	return torch.utils.data.dataloader.default_collate(batch)
	"""
	seqs, signals, targets, events, sims = [],[],[],[],[]
	aligns = []

	for rf in batch:
		if rf is not None:
			# the expanded data itself is a list
			feats, labels = rf
			
			seqs.extend([[nc_dic[c] for c in f[0]] for f in feats])
			
			# add the simulation normalization here
			## uncomment if the simualtion informaiton is intended to be used.
			#sims.extend([sim_seq_singal(f[0]) for f in feats])

			signals.extend([f[1] for f in feats])
			events.extend([[e for e in f[3]] for f in feats])

			# replace the methylated ones with the simulator
			"""
			label_count = labels.count(0)
			if label_count > 0:
				assert(label_count == len(labels))
				events.extend([sim_seq_singal(f[0][5:-5]) for f in feats])
			else:
				events.extend([[e for e in f[3][5:-5]] for f in feats])
			"""	
			
			targets.extend(labels)

			# add the alignment information, not send  
			aligns.extend([f[2] for f in feats])

	
	# unstack motif-level signal samples
	seqs    = torch.tensor(seqs, dtype=torch.long)
	signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
	events  = torch.tensor(events, dtype=torch.float32) #.permute(0,2,1)
	targets = torch.tensor(targets, dtype=torch.int16)

	# sim data, chech whehter this information is useful
	sims = None #torch.tensor(sims, dtype=torch.float32)

	return (seqs, signals, events, sims, aligns), targets

# 2020/05/04 loading from two folds
def load_from_2folds(meth_fold_path, pcr_fold_path, num_worker, b_size=20, motif=["CG"], m_shift=0, w_len=17, train_test_split=(0.8, 0.1), data_balance=False):

	# fix the random seed
	random_seed = 123
	torch.manual_seed(random_seed)
	print(" |- Setting random seed of [%d] before loading the data." %(random_seed))

	train_split, test_split = train_test_split
	# data loading
	print(" |- Loading data ...")

	data_set1  = MethylDataSet(get_fast5s(meth_fold_path), 1, motif, m_shift, w_len)
	### split the data
	n_train, n_test = int(len(data_set1) * train_split), int(len(data_set1) * test_split)
	print("* Rough number of [1]-methylated: %d" %(n_train))
	n_val = len(data_set1) - n_train - n_test
	p_train, p_test, p_val = random_split(data_set1, [n_train, n_test, n_val])

	# non-equal number matching ones
	data_set2  = MethylDataSet(get_fast5s(pcr_fold_path),  0, motif, m_shift, w_len)

	# 20210120 added
	if len(data_set2) > len(data_set1) and data_balance == True:
		indices = torch.randperm(len(data_set2))[:len(data_set1)]
		data_set2= torch.utils.data.Subset(data_set2, indices)
		#data_set2 = data_set2[indices]

	### split the data
	n_train, n_test = int(len(data_set2) * train_split), int(len(data_set2) * test_split)
	print("* Rough number of [0]-methylated: %d" %(n_train))
	n_val = len(data_set2) - n_train - n_test
	n_train, n_test, n_val = random_split(data_set2, [n_train, n_test, n_val])

	train_set = ConcatDataset([p_train, n_train])
	test_set  = ConcatDataset([p_test,  n_test])
	val_set   = ConcatDataset([p_val,   n_val])

	train_generator = DataLoader(train_set, batch_size=b_size, shuffle=True, num_workers=num_worker, collate_fn=collate_fn)
	test_generator =  DataLoader(test_set,  batch_size=b_size, shuffle=False,num_workers=num_worker, collate_fn=collate_fn)
	val_generator =   DataLoader(val_set,   batch_size=b_size, shuffle=False,num_workers=num_worker, collate_fn=collate_fn)

	return (train_generator, test_generator, val_generator)

# 20200715, add function for the data similarity analysis
def data_analysis(meth_fold_path, pcr_fold_path, num_worker, b_size=20, motif=["CG"], m_shift=0, w_len=17):
	
	# fix the random seed
	random_seed = 123
	torch.manual_seed(random_seed)
	print(" |- Setting random seed of [%d] before loading the data." %(random_seed))
	print(" |- Loading data for basic analysis ...")

	# data loading
	print(" |- Loading data ...")
	data_set1  = MethylDataSet(get_fast5s(meth_fold_path), 1, motif, m_shift, w_len)
	print("* Batch number of [1]-methylated: %d" %(len(data_set1)))
	meth_generator = DataLoader(data_set1, batch_size=b_size, shuffle=False, num_workers=num_worker, collate_fn=collate_fn)
	
	data_set2  = MethylDataSet(get_fast5s(pcr_fold_path),  0, motif, m_shift, w_len)
	print("* Batch number of [0]-methylated: %d" %(len(data_set2)))
	unMeth_generator = DataLoader(data_set2, batch_size=b_size, shuffle=False, num_workers=num_worker, collate_fn=collate_fn)

	seq_meth, seq_unMeth = [],[]
	#loading all the sequences of the data
	for i, data in enumerate(meth_generator, 0):
		inputs, labels = data
		if len(labels) == 0: 
			skip_batch += 1
			continue
		seqs = inputs[0].data.cpu().numpy()
		seq_meth.extend([ "".join([str(x) for x in s]) for s in seqs])

	seq_meth_uniq = set(seq_meth)
	print("+ Totally get %d meth sequences." %(len(seq_meth)))
	print("+ Totally get %d unique meth sequences." %(len(seq_meth_uniq)))

	for i, data in enumerate(unMeth_generator, 0):
		inputs, labels = data
		if len(labels) == 0: 
			skip_batch += 1
			continue
		seqs = inputs[0].data.cpu().numpy()
		seq_unMeth.extend(["".join([str(x) for x in s]) for s in seqs])

	seq_unMeth_uniq = set(seq_unMeth)
	print("+ Totally get %d unMeth sequences." %(len(seq_unMeth)))
	print("+ Totally get %d unique unMeth sequences." %(len(seq_unMeth_uniq)))

	seq_overlap = seq_meth_uniq.intersection(seq_unMeth_uniq)
	print("* Overalpped sequence in the training data is [%d]" %(len(seq_overlap)))

# 20200520, load data from single file
def load_from_single_folds(fold_path, num_worker, label=1, b_size=20, motif=["CG"], m_shift=0, w_len=17):

	# fix the random seed
	random_seed = 123
	torch.manual_seed(random_seed)

	print(" |- Loading single fold data with label-[%d] ..." %(label))

	data_set  = MethylDataSet(get_fast5s(fold_path), label, motif, m_shift, w_len)
	generator = DataLoader(data_set, batch_size=b_size, shuffle=False, num_workers=num_worker, collate_fn=collate_fn)

	return generator


if __name__ == "__main__":

	fold_path = "/home/yaozhong/working/2_nanopore/methylation/data/dev/ecoli_er2925.pcr_MSssI.r9.timp.061716.fast5"
	
	file_list = get_fast5s(fold_path)
	read_data_set = MethylDataSet(file_list, 1)

	# batch_size is the maximum number of reads in each batch
	meth_generator = DataLoader(read_data_set, batch_size=50, shuffle=False, num_workers=1, collate_fn=collate_fn)

	next(iter(meth_generator))








