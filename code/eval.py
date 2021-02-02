#!/usr/bin/python

import torch
import torch.nn.functional as F
import time,tqdm,os,math
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
import numpy as np
from model.siamese_triplet import triplet_data_gen
from dataProcess.visual_util import tSNE_plot
from dataProcess.data_loading import *
from dataProcess.ref_util import *
from torchsummary import summary
import time

cn_dic = {'0':'A', '1':'T', '2':'G','3':'C', '4':'N'}

def get_group_pred(mList):

	cov = len(mList)
	meth = mList.count(1)/cov
	unMeth = mList.count(0)/cov

	# current can achieved best group results
	if meth >= 0.1:
		return 1
	else:
		return 0
	
	"""
	if meth > unMeth:
		return 1
	else:
		return 0
	"""
	

# 20200612 group evaluation
# 20200722 confirmed negative strand location count
def group_eval(align_list, output_list, gold_list, label=1, threshold=5):

	position_dic, gold_dic = {}, {}

	for (a, pred, gold) in zip(align_list, output_list, gold_list):

		# seperate group the values, assume only two types of label
		if gold != label:
			continue

		if a[3] == "-":
			## chrome name,  start, motif_relative_location, align_strand, strand_len
			## using the non-strand specific ones.
			align = (a[0], "-", a[1]+a[4]-a[2]-1) #this position seems not totally corrected.
		else:
			align = (a[0], "+", a[1]+a[2])
	
		# prediction
		if align not in position_dic.keys():
			position_dic[align] = [pred]
		else:
			position_dic[align].append(pred)
	
	#print("-x- Total predictions %d" %(len(output_list)))
	print(" |- Total [%d] grouped %s-label of the sample" %(len(position_dic.keys()), str(label)))

	if len(position_dic.keys())== 0:
		return [],[]

	depth = [len(position_dic[k]) for k in position_dic.keys()]
	print("    -> min_depth=%d, max_depth=%d, mean_depth=%f" %(min(depth), max(depth),np.mean(depth)))

	## evaulations
	gold_list, pred_list = [],[]

	#for pos in tqdm.tqdm(position_dic.keys(), total=len(position_dic.keys())):
	for pos in position_dic.keys():
		if len(position_dic[pos]) < threshold:
			continue

		pred_list.append(get_group_pred(position_dic[pos]))
		gold_list.append(label)

	return pred_list, gold_list


def binary_acc(preds, y):
    preds = torch.argmax(preds, dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum()
    return acc


# evluation metrics
def evaluate_signal(net, criterion, device, val_generator):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	net.eval()
	with torch.no_grad():
		for i, val_data in enumerate(val_generator, 0):
			inputs, labels = val_data

			if len(labels) == 0: continue
			n_sample += len(labels)

			outputs = net(inputs[1].to(device))
			batch_loss = criterion(outputs, labels.to(device).long())
			val_loss += batch_loss.item()*len(labels)
			accuracy += binary_acc(outputs, labels.to(device).long()).item()

			pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
			gold_list.extend(labels.numpy())

		valid_loss = val_loss/n_sample
		accuracy   = accuracy/n_sample

		# Calculation of AUC
		auc_score = roc_auc_score(gold_list, pred_list)
		tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
		precision, recall = tp/(tp+fp), tp/(tp+fn)
		sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

	return	valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity

def evaluate_unseg_seq(net, criterion, device, val_generator, useSEQ):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	net.eval()
	with torch.no_grad():
		for i, val_data in enumerate(val_generator, 0):
		#for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
			inputs, labels = val_data

			if len(labels) == 0: continue
			n_sample += len(labels)

			seq_event = inputs[1]
			seq_event = seq_event.reshape(seq_event.shape[0], -1, 7)

			outputs = net(seq_event.to(device))
			batch_loss = criterion(outputs, labels.to(device).long())
			val_loss += batch_loss.item()*len(labels)
			accuracy += binary_acc(outputs, labels.to(device).long()).item()

			pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
			gold_list.extend(labels.numpy())

		valid_loss = val_loss/n_sample
		accuracy   = accuracy/n_sample

		# Calculation of AUC
		auc_score = roc_auc_score(gold_list, pred_list)
		tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
		precision, recall = tp/(tp+fp), tp/(tp+fn)
		sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
		print("== Total test input-instance [%d]" %(n_sample))
		
	return	valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity


# add the function of the group evaluation
## to do add the group evaluation.
def evaluate_event(net, criterion, device, val_generator, useSEQ, use_sim, generator=None, permute=False, vis=False, siamese=False):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	# group evaluation 
	align_list, output_list = [], []

	total_time = 0

	net.eval()
	with torch.no_grad():

		for i, val_data in enumerate(val_generator, 0):
		#for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
			inputs, labels = val_data

			if len(labels) == 0: continue
			n_sample += len(labels)

			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float().to(device), inputs[2].to(device)) , -1)

				if use_sim:
					#seq_event = torch.cat((seq.float(), inputs[2]-inputs[3]), -1)
					#seq_event = torch.cat((seq.float(), inputs[2][:,:,0:2]-inputs[3][:,:,0:2]), -1)
					seq_event = torch.cat((seq.float().to(device), inputs[2].to(device), inputs[2].to(device)-inputs[3].to(device)), -1)
				if generator != None:
					fake_signal = generator(seq.to(device).float())
					seq_event = torch.cat((seq.float().to(device), inputs[2].to(device), fake_signal.to(device)), -1)
			else:
				seq_event = inputs[2]

			if permute:
				seq_event = seq_event.permute(0,2,1)
			
			# decoding time	
			if siamese == True:
				outputs = net(seq_event.to(device), None)
			else:
				outputs = net(seq_event.to(device))

			align_list.extend(inputs[4])

			# evaluation
			batch_loss = criterion(outputs, labels.to(device).long())
			val_loss += batch_loss.item()*len(labels)
			accuracy += binary_acc(outputs, labels.to(device).long()).item()

			pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
			# added cpu() here
			gold_list.extend(labels.cpu().numpy())

		valid_loss = val_loss/n_sample
		accuracy   = accuracy/n_sample

		# Calculation of AUC
		auc_score = roc_auc_score(gold_list, pred_list)
		tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
		precision, recall = tp/(tp+fp), tp/(tp+fn)
		sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

		acc = (tp + tn) / (tp+tn+fp+fn)
		f1  = 2*tp / (2*tp + fp + fn)
		#mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
		mcc = matthews_corrcoef(gold_list, pred_list)

		print(" @ <Read-level evaluation>")
		print(" |* Total test input-instance [%d]" %(n_sample))
		print(" |= tn=%d, fp=%d, fn=%d, tp=%d" %(tn, fp, fn, tp))
		print(" |= AUC=%.4f, Precision=%.4f, Recall=%.4f" %(auc_score, precision, recall))
		print(" |= ACC=%.4f, F1=%.4f, MCC=%.4f" %(acc, f1, mcc))

	# Group evaluation
	cov_thres = 1
	print(" @ <Group-level evaluation with >=%d >" %(cov_thres))

	group_gold_list, group_pred_list = [], []
	p,g = group_eval(align_list, pred_list, gold_list, 1, cov_thres)
	group_pred_list.extend(p)
	group_gold_list.extend(g)
	
	p,g = group_eval(align_list, pred_list, gold_list, 0, cov_thres)
	group_pred_list.extend(p)
	group_gold_list.extend(g)

	group_auc_score = roc_auc_score(group_gold_list, group_pred_list)
	group_tn, group_fp, group_fn, group_tp = confusion_matrix(group_gold_list, group_pred_list).ravel()  
	group_precision, group_recall = group_tp/(group_tp+group_fp), group_tp/(group_tp+group_fn)
	group_sensitivity, group_specificity = group_tp/(group_tp+group_fn), group_tn/(group_tn+group_fp)

	group_acc = (group_tp + group_tn) / (group_tp+group_tn+group_fp+group_fn)
	group_f1  = 2*group_tp / (2*group_tp + group_fp + group_fn)
	#group_mcc = math.sqrt((group_tp*group_tn - group_fp*group_fn)**2 / (group_tp+group_fp)*(group_tp+group_fn)*(group_tn+group_fp)*(group_tn+group_fn))
	group_mcc = matthews_corrcoef(group_gold_list, group_pred_list)

	print(" |= group_tn=%d, group_fp=%d, group_fn=%d, group_tp=%d" %(group_tn, group_fp, group_fn, group_tp))
	print(" |= group_AUC=%.4f, group_Precision=%.4f, group_Recall=%.4f" %(group_auc_score, group_precision, group_recall))
	print(" |= group_ACC=%.4f, group_F1=%.4f, group_MCC=%.4f" %(group_acc, group_f1, group_mcc))

	# visualization for analysis, not complete
	if vis:
		index0 = [i for i, e in enumerate(gold_list) if e == 0]
		index1 = [i for i, e in enumerate(gold_list) if e == 1]
		align0, align1 = set([align_list[i] for i in index0]), set([align_list[i] for i in index1])

		print("-- overlap regions for both label")
		align_overlap = list(align1.intersection(align0))
		#if len(align_overlap) > 0:
		#	idx = align_overlap()

	return	valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity


# added two seperate
def evaluate_event_for_GANsim(net, criterion, device, val_generator, useSEQ, use_sim, generator_m, generator_um, vis=False):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	# group evaluation 
	align_list, output_list = [], []

	net.eval()
	with torch.no_grad():

		for i, val_data in enumerate(val_generator, 0):
		#for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
			inputs, labels = val_data

			if len(labels) == 0: continue
			n_sample += len(labels)

			# using simulator 
			seq = F.one_hot(inputs[0]) #.permute(0,2,1)

			# replacement of the original signal with 
			label_count = (labels==0).sum()
			if label_count > 0:
				# replace the singal with the methylated simualtor
				fake_signal = generator_um(seq[labels==0].to(device).float())
				seq_event_0 = torch.cat((seq[labels==0].float().to(device), fake_signal.to(device)), -1)
				#seq_event_0 = torch.cat((seq[(labels==0)].float().to(device), inputs[2][(labels==0),:,0:1].to(device)), -1)
				labels_0 = labels[(labels==0)]

				labels_0 = labels[labels==0]

			if len(labels) > label_count:
				# using the 
				fake_signal = generator_m(seq[~(labels==0)].to(device).float())
				seq_event_1 = torch.cat((seq[~(labels==0)].float().to(device), fake_signal.to(device)), -1)
				#seq_event_1 = torch.cat((seq[~(labels==0)].float().to(device), inputs[2][~(labels==0),:,0:1].to(device)), -1)
				labels_1 = labels[~(labels==0)]

			if label_count == len(labels):
				seq_event = seq_event_0
			elif label_count == 0:
				seq_event = seq_event_1
			else:
				seq_event = torch.cat((seq_event_1, seq_event_0), 0)
				labels = torch.cat((labels_1, labels_0), 0)


			"""
			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float().to(device), inputs[2].to(device)) , -1)

				if use_sim:
					#seq_event = torch.cat((seq.float(), inputs[2]-inputs[3]), -1)
					#seq_event = torch.cat((seq.float(), inputs[2][:,:,0:2]-inputs[3][:,:,0:2]), -1)
					seq_event = torch.cat((seq.float().to(device), inputs[2].to(device), inputs[2].to(device)-inputs[3].to(device)), -1)
				if generator != None:
					fake_signal = generator(seq.to(device).float())
					#seq_event = torch.cat((seq.float().to(device), inputs[2].to(device)-fake_signal.to(device)), -1)
					seq_event = torch.cat((seq.float().to(device), inputs[2].to(device), fake_signal.to(device)), -1)
			else:
				seq_event = inputs[2]
			"""

			outputs = net(seq_event.to(device))
			align_list.extend(inputs[4])

			# evaluation
			batch_loss = criterion(outputs, labels.to(device).long())
			val_loss += batch_loss.item()*len(labels)
			accuracy += binary_acc(outputs, labels.to(device).long()).item()

			pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
			# added cpu() here
			gold_list.extend(labels.cpu().numpy())

		valid_loss = val_loss/n_sample
		accuracy   = accuracy/n_sample

		# Calculation of AUC
		auc_score = roc_auc_score(gold_list, pred_list)
		tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
		precision, recall = tp/(tp+fp), tp/(tp+fn)
		sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
		print(" @ <Read-level evaluation>")
		print(" |* Total test input-instance [%d]" %(n_sample))
		print(" |= tn=%d, fp=%d, fn=%d, tp=%d" %(tn, fp, fn, tp))
		print(" |= AUC=%f, Precision=%f, Recall=%f" %(auc_score, precision, recall))

	# Group evaluation
	print(" @ <Group-level evaluation>")
	group_gold_list, group_pred_list = [], []
	p,g = group_eval(align_list, pred_list, gold_list, 1)
	group_pred_list.extend(p)
	group_gold_list.extend(g)
	

	p,g = group_eval(align_list, pred_list, gold_list, 0)
	group_pred_list.extend(p)
	group_gold_list.extend(g)

	group_auc_score = roc_auc_score(group_gold_list, group_pred_list)
	group_tn, group_fp, group_fn, group_tp = confusion_matrix(group_gold_list, group_pred_list).ravel()  
	group_precision, group_recall = group_tp/(group_tp+group_fp), group_tp/(group_tp+group_fn)
	group_sensitivity, group_specificity = group_tp/(group_tp+group_fn), group_tn/(group_tn+group_fp)

	group_acc = (group_tp + group_tn) / (group_tp+group_tn+group_fp+group_fn)
	group_f1  = 2*group_tp / (2*group_tp + group_fp + group_fn)
	#group_mcc = math.sqrt((group_tp*group_tn - group_fp*group_fn)**2 / (group_tp+group_fp)*(group_tp+group_fn)*(group_tn+group_fp)*(group_tn+group_fn))
	group_mcc = matthews_corrcoef(group_gold_list, group_pred_list)

	print(" |= group_tn=%d, group_fp=%d, group_fn=%d, group_tp=%d" %(group_tn, group_fp, group_fn, group_tp))
	print(" |= group_AUC=%.4f, group_Precision=%.4f, group_Recall=%.4f" %(group_auc_score, group_precision, group_recall))
	print(" |= group_ACC=%.4f, group_F1=%.4f, group_MCC=%.4f" %(group_acc, group_f1, group_mcc))

	# visualization for analysis, not complete
	if vis:
		index0 = [i for i, e in enumerate(gold_list) if e == 0]
		index1 = [i for i, e in enumerate(gold_list) if e == 1]
		align0, align1 = set([align_list[i] for i in index0]), set([align_list[i] for i in index1])

		print("-- overlap regions for both label")
		align_overlap = list(align1.intersection(align0))
		#if len(align_overlap) > 0:
		#	idx = align_overlap()

	return	valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity



def evaluate_single(net, criterion, device, val_generator, useSEQ):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	net.eval()
	with torch.no_grad():
		#for i, val_data in enumerate(val_generator, 0):
		for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
			inputs, labels = val_data

			if len(labels) == 0: continue
			n_sample += len(labels)

			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float(), inputs[2]) , -1)
			else:
				seq_event = inputs[2]

			outputs = net(seq_event.to(device))
			batch_loss = criterion(outputs, labels.to(device).long())
			val_loss += batch_loss.item()*len(labels)
			accuracy += binary_acc(outputs, labels.to(device).long()).item()

			pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
			gold_list.extend(labels.numpy())

		valid_loss = val_loss/n_sample
		accuracy   = accuracy/n_sample

		# Calculation of AUC
		#auc_score = roc_auc_score(gold_list, pred_list)
		#tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
		#precision, recall = tp/(tp+fp), tp/(tp+fn)
		#sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
		print("== Total test input-instance [%d]" %(n_sample))
		
	return	valid_loss, accuracy

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

# 20210105, refine the file output
def evaluate_print(net, criterion, device, val_generator, useSEQ,  ref_genome, output_file):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	total_decode_time = 0
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	f=open(output_file, "w")

	if ref_genome == None:
		print("Please specify the reference genome used for the read pre-processing !")
		exit(-1)
	else:
		chrom2len = get_contig2len(ref_genome)

	net.eval()
	with torch.no_grad():
		for i, val_data in enumerate(val_generator, 0):
		#for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
			inputs, labels = val_data

			# the labels here will not be used in the evaluation
			if len(labels) == 0: continue
			n_sample += len(labels)

			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float(), inputs[2]) , -1)
			else:
				seq_event = inputs[2]

			align_info = inputs[4]

			start=time.time()
			output = net(seq_event.to(device))
			decode_time = time.time()-start
			total_decode_time += decode_time

			pred_list.extend(torch.argmax(output, dim=1).cpu().numpy())
			gold_list.extend(labels.numpy())
			accuracy += binary_acc(output, labels.to(device).long()).item()

			#output = F.softmax(output).to("cpu").detach().numpy().copy()

			# special transform for the bert_plus
			output = (output.to("cpu").detach().numpy().copy() + 1)/2  # if tanh is used

			# preparing the output format
			#(chrom, chrom_start, motif_in_read, align_strand, read_length, readname, strand)
			#a_info = (align_info[0], align_info[1], loci, align_info[3], len(seq_event), align_info[4], align_info[2])
			
			for j in range(len(align_info)):

				chr_name, strand, read_id, read_strand =  align_info[j][0], align_info[j][3], align_info[j][5], align_info[j][6]

				# position's calcuation: pos, and pos_in_strand, to be test on the real dataset
				if strand == "+":
					pos = align_info[j][1]+align_info[j][2]
					pos_in_strand = pos
				else:
					pos =  align_info[j][1] + align_info[j][4] - align_info[j][2] - 1 
					# align = (a[1]+a[4]-a[2]-1)
					pos_in_strand = chrom2len[chr_name] - pos -1

				f.write("%s\t%s\t%s\t%s\t%s\t%s\t" %(chr_name, pos, strand, pos_in_strand, read_id,read_strand))
				f.write("\t".join([str(score) for score in output[j]]))

				if output[j][0] > output[j][1]:
					f.write("\t0\t")
				else:
					f.write("\t1\t")

				f.write("".join([cn_dic[str(x)] for x in inputs[0][j].tolist()]))
				f.write("\n")
		
		# if needed put into the pandas for the data processing	
		#print("== Total test input-instance [%d]" %(n_sample))
		f.close()

		print("ACC is", accuracy/n_sample)
		print("*Deocding time is ", total_decode_time)



def evaluate_bert(net, criterion, device, val_generator, useSEQ, use_sim):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	# group evaluation 
	align_list, output_list = [], []

	net.eval()
	with torch.no_grad():
		for i, val_data in enumerate(val_generator, 0):
			inputs, labels = val_data

			if len(labels) == 0: continue
			n_sample += len(labels)

			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				if use_sim:
					seq_event = torch.cat((seq.float(), inputs[2]-inputs[3]) , -1)
				else:
					seq_event = torch.cat((seq.float(), inputs[2]) , -1)
			else:
				seq_event = inputs[2]

			outputs = net(seq_event.to(device), None)
			align_list.extend(inputs[4])

			batch_loss = criterion(outputs, labels.to(device).long())
			#batch_loss = criterion(outputs, labels.to(device).unsqueeze(1).as_type(outputs))
			val_loss += batch_loss.item()*len(labels)
			accuracy += binary_acc(outputs, labels.to(device).long()).item()
			#accuracy += binary_acc(outputs, labels.to(device).unsqueeze(1).as_type(outputs)).item()

			pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
			#y_test_pred = torch.sigmoid(outputs)
			#y_pred_tag = torch.round(y_test_pred)		
			#pred_list.extend(y_pred_tag.cpu().numpy())
			
			gold_list.extend(labels.cpu().numpy())

		valid_loss = val_loss/n_sample
		accuracy   = accuracy/n_sample

		# Calculation of AUC
		auc_score = roc_auc_score(gold_list, pred_list)
		tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
		precision, recall = tp/(tp+fp), tp/(tp+fn)
		sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

		acc = (tp + tn) / (tp+tn+fp+fn)
		f1  = 2*tp / (2*tp + fp + fn)
		#mcc = math.sqrt((tp*tn - fp*fn)**2 /(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
		mcc = matthews_corrcoef(gold_list, pred_list)

		print(" @ <Read-level evaluation>")
		print(" |* Total test input-instance [%d]" %(n_sample))
		print(" |= tn=%d, fp=%d, fn=%d, tp=%d" %(tn, fp, fn, tp))
		print(" |= AUC=%.4f, Precision=%.4f, Recall=%.4f" %(auc_score, precision, recall))
		print(" |= ACC=%.4f, F1=%.4f, MCC=%.4f" %(acc, f1, mcc))
	
	#Group evluation
	cov_thres = 1
	print(" @ <Group-level evaluation with >=%d >" %(cov_thres))

	group_gold_list, group_pred_list = [], []
	p,g = group_eval(align_list, pred_list, gold_list, 1, cov_thres)
	group_pred_list.extend(p)
	group_gold_list.extend(g)
	
	p,g = group_eval(align_list, pred_list, gold_list, 0, cov_thres)
	group_pred_list.extend(p)
	group_gold_list.extend(g)

	group_auc_score = roc_auc_score(group_gold_list, group_pred_list)
	group_tn, group_fp, group_fn, group_tp = confusion_matrix(group_gold_list, group_pred_list).ravel()  
	group_precision, group_recall = group_tp/(group_tp+group_fp), group_tp/(group_tp+group_fn)
	group_sensitivity, group_specificity = group_tp/(group_tp+group_fn), group_tn/(group_tn+group_fp)

	group_acc = (group_tp + group_tn) / (group_tp+group_tn+group_fp+group_fn)
	group_f1  = 2*group_tp / (2*group_tp + group_fp + group_fn)
	#group_mcc = math.sqrt((group_tp*group_tn - group_fp*group_fn)**2 / (group_tp+group_fp)*(group_tp+group_fn)*(group_tn+group_fp)*(group_tn+group_fn))
	group_mcc = matthews_corrcoef(group_gold_list, group_pred_list)

	print(" |= group_tn=%d, group_fp=%d, group_fn=%d, group_tp=%d" %(group_tn, group_fp, group_fn, group_tp))
	print(" |= group_AUC=%.4f, group_Precision=%.4f, group_Recall=%.4f" %(group_auc_score, group_precision, group_recall))
	print(" |= group_ACC=%.4f, group_F1=%.4f, group_MCC=%.4f" %(group_acc, group_f1, group_mcc))
	
	return	valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity


## 2020/05/13 added for the hierBERT
def evaluate_hierbert(net, criterion, device, val_generator):

	val_loss, accuracy, n_sample = 0, 0, 0
	pred_list, gold_list = [], []

	net.eval()
	with torch.no_grad():
		for i, val_data in enumerate(val_generator, 0):
			inputs, labels = val_data

			if len(labels) == 0: continue
			n_sample += len(labels)

			seq_event = (inputs[0], inputs[2])

			outputs = net((seq_event[0].to(device), seq_event[1].to(device)))

			batch_loss = criterion(outputs, labels.to(device).long())
			val_loss += batch_loss.item()*len(labels)
			accuracy += binary_acc(outputs, labels.to(device).long()).item()

			pred_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
			gold_list.extend(labels.numpy())

		valid_loss = val_loss/n_sample
		accuracy   = accuracy/n_sample

		# Calculation of AUC
		auc_score = roc_auc_score(gold_list, pred_list)
		tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
		precision, recall = tp/(tp+fp), tp/(tp+fn)
		sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
		print("== Total test input-instance [%d]" %(n_sample))
		
	return	valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity

# 2020/08/04
def gen_embeddings(net, input_vis, device, batch=128):
	
	net = net.to(device)

	with torch.no_grad():

		nbatch = int(len(input_vis)/batch)
		embed_list = []

		# bach like prediction
		for bi in range(nbatch):
			embed_singal_vis = net.get_embedding(input_vis[bi*batch:(bi+1)*batch].to(device))
			embed_list.append(embed_singal_vis)

		if (nbatch*batch < len(input_vis)):
			embed_singal_vis = net.get_embedding(input_vis[nbatch*batch:].to(device))
			embed_list.append(embed_singal_vis)

		embed_vis = torch.cat(embed_list, dim=0)
		return embed_vis.to("cpu")


def gen_embeddings2(embed, input_vis, device, batch=128):

	with torch.no_grad():

		nbatch = int(len(input_vis)/batch)
		embed_list = []

		# bach like prediction
		for bi in range(nbatch):
			embed_singal_vis = embed(input_vis[bi*batch:(bi+1)*batch].to(device))
			embed_list.append(embed_singal_vis)

		if (nbatch*batch < len(input_vis)):
			embed_singal_vis = embed(input_vis[nbatch*batch:].to(device))
			embed_list.append(embed_singal_vis)

		embed_vis = torch.cat(embed_list, dim=0)
		return embed_vis.to("cpu")


# evaluation of the triplet loss, take care of this part
def evaluate_validate_triplet(net, criterion, device, val_generator, useSEQ, use_sim, epoch, generator=None, vis=False, siamese=False, do_tSNE_plot=True, figure_prefix="", svm_model=None):

	valid_loss,n_sample = 0, 0

	# group evaluation 
	align_list, output_list = [], []

	net.eval()
	with torch.no_grad():

		seq_group_data = {}
		input_vis, signal_vis, label_vis =[], [], []

		# loading
		for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
		#for i, val_data in enumerate(val_generator, 0):
			inputs, labels = val_data
			if len(labels) == 0: continue

			for bi in range(len(inputs[0])):
				# chekcing this function
				s = "".join([str(x) for x in inputs[0][bi].data.numpy()])
				if s not in seq_group_data.keys():
					seq_group_data[s]=[[],[]]
				idx = labels[bi].item()
				seq_group_data[s][idx].append(inputs[2][bi])

			# used only for the visualization
			if do_tSNE_plot:
				signal_vis.append(inputs[2][:,:,0])
				label_vis.append(labels)

				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float(), inputs[2]) , -1)
				#seq_event = inputs[2]
				input_vis.append(seq_event)
		
		if do_tSNE_plot:
			input_vis  = torch.cat(input_vis, dim=0)
			signal_vis = torch.cat(signal_vis, dim=0)
			label_vis  = torch.cat(label_vis,  dim=0)

			# checking the plot path
			if not os.path.exists("../experiment/figures/siamese/tunning/" + figure_prefix):
				os.mkdir("../experiment/figures/siamese/tunning/" + figure_prefix)

			print(" + generate original t-SNE visualization for %d samples..." %(len(label_vis)))
			tSNE_plot(signal_vis.cpu().numpy(), label_vis.cpu().numpy(), "../experiment/figures/siamese/tunning/" + figure_prefix + "/3D_origin_signal_Epoch-"+str(epoch)+ "_validate_tSNE_5000.png", 5000, 3)
			tSNE_plot(signal_vis.cpu().numpy(), label_vis.cpu().numpy(), "../experiment/figures/siamese/tunning/" + figure_prefix + "/2D_origin_signal_Epoch-"+str(epoch)+ "_validate_tSNE_5000.png", 5000, 2)
			
			# generate embed list
			embed_vis = gen_embeddings(net, input_vis, device, batch=32)
			tSNE_plot(embed_vis.cpu().numpy(), label_vis.cpu().numpy(), "../experiment/figures/siamese/tunning/" +  figure_prefix + "/3D_embed_signal_Epoch-"+str(epoch)+"_validate_tSNE_5000.png", 5000, 3)
			tSNE_plot(embed_vis.cpu().numpy(), label_vis.cpu().numpy(), "../experiment/figures/siamese/tunning/" +  figure_prefix + "/2D_embed_signal_Epoch-"+str(epoch)+"_validate_tSNE_5000.png", 5000, 2)

			## direct evluation of the SVM, RF shallow model
			if svm_model is not None:
				print("- Evaluating shallow model [SVM]...")
				t = svm_model.predict(embed_vis).ravel()
				
				auc_score = roc_auc_score(label_vis, t)
				tn, fp, fn, tp = confusion_matrix(label_vis, t).ravel()  

				precision, recall = tp/(tp+fp), tp/(tp+fn)
				sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

				acc = (tp + tn) / (tp+tn+fp+fn)
				f1  = 2*tp / (2*tp + fp + fn)
				#mcc = math.sqrt((tp*tn - fp*fn)**2 / (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
				mcc = matthews_corrcoef(gold_list, pred_list)

				print(" @ <SVM Read-level evaluation>")
				print(" |= tn=%d, fp=%d, fn=%d, tp=%d" %(tn, fp, fn, tp))
				print(" |= AUC=%.4f, Precision=%.4f, Recall=%.4f" %(auc_score, precision, recall))
				print(" |= ACC=%.4f, F1=%.4f, MCC=%.4f" %(acc, f1, mcc))

				

		#  all loading for 
		#seq_group_dataloder = triplet_data_gen(seq_group_data, device)
		#n_sample = len(seq_group_dataloder)

		# new batch loading 
		print(" + [inside]: Generate dataloading ...")
		seq_group_dataset = seqTripletSampleSet(seq_group_data, device)
		seq_group_dataloder = DataLoader(seq_group_dataset, batch_size=32,collate_fn=my_collate2)
		n_sample = 0

		# triplet data evaluation	
		for i, pair_data in tqdm.tqdm(enumerate(seq_group_dataloder, 0), total=len(seq_group_dataloder)):
			if(pair_data is None): continue
			X1, X2, X3 = pair_data[0], pair_data[1], pair_data[2]

			o1,o2,o3 = net(X1.to(device), X2.to(device), X3.to(device))
			batch_loss = criterion(o1,o2,o3)
			valid_loss += batch_loss.item()*len(X1)
			n_sample += len(X1)
			
		valid_loss = valid_loss/n_sample

	return	valid_loss


# evaluate shallow model 
def evaluate_validate_shallow(net, device, val_generator, s_model):

	print("@ enter evaluation mode ...")

	net.eval()
	with torch.no_grad():
		input_vis, signal_vis, label_vis =[], [], []
		for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
		#for i, val_data in enumerate(val_generator, 0):
			inputs, labels = val_data
			if len(labels) == 0: continue

			signal_vis.append(inputs[2][:,:,0])
			label_vis.append(labels)

			seq = F.one_hot(inputs[0]) #.permute(0,2,1)
			#seq_event = torch.cat((seq.float(), inputs[2][:,:,0:2]) , -1)
			seq_event = torch.cat((seq.float(), inputs[2]) , -1)
			#seq_event = inputs[2]
			
			input_vis.append(seq_event)
		
		input_vis  = torch.cat(input_vis, dim=0)
		signal_vis = torch.cat(signal_vis, dim=0)
		label_vis  = torch.cat(label_vis,  dim=0)

		embed_vis = gen_embeddings2(net, input_vis, device, batch=32)

		## direct evluation of the SVM, RF shallow model
		print("- Evaluating shallow model with Embedding results...")
		t = s_model.predict(embed_vis).ravel()

		auc_score = roc_auc_score(label_vis, t)
		tn, fp, fn, tp = confusion_matrix(label_vis, t).ravel()     

		precision, recall = tp/(tp+fp), tp/(tp+fn)
		sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

		acc = (tp + tn) / (tp+tn+fp+fn)
		f1  = 2*tp / (2*tp + fp + fn)
		#mcc = math.sqrt( (tp*tn - fp*fn)**2 / (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
		mcc = matthews_corrcoef(gold_list, pred_list)

		print(" @ <Shallow model Read-level evaluation>")
		print(" |= tn=%d, fp=%d, fn=%d, tp=%d" %(tn, fp, fn, tp))
		print(" |= AUC=%.4f, Precision=%.4f, Recall=%.4f" %(auc_score, precision, recall))
		print(" |= ACC=%.4f, F1=%.4f, MCC=%.4f" %(acc, f1, mcc))


def evaluate_validate_shallow_simple(val_generator, s_model):

	print("@ enter evaluation mode ...")

	signal_vis, label_vis =[], []
	for i, val_data in tqdm.tqdm(enumerate(val_generator, 0), total=len(val_generator)):
		inputs, labels = val_data
		if len(labels) == 0: continue

		signal_vis.append(inputs[2][:,:,0])
		label_vis.append(labels)

	print("+ Merging tensors ...")
		
	signal_vis = torch.cat(signal_vis, dim=0)
	label_vis  = torch.cat(label_vis,  dim=0)


	## direct evluation of the SVM, RF shallow model
	print("- Evaluating shallow model with Embedding results...")
	t = s_model.predict(signal_vis).ravel()

	auc_score = roc_auc_score(label_vis, t)
	tn, fp, fn, tp = confusion_matrix(label_vis, t).ravel()     

	precision, recall = tp/(tp+fp), tp/(tp+fn)
	sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

	acc = (tp + tn) / (tp+tn+fp+fn)
	f1  = 2*tp / (2*tp + fp + fn)
	#mcc = math.sqrt((tp*tn - fp*fn)**2 / (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
	mcc = matthews_corrcoef(label_vis, t)

	print(" @ <Shallow model Read-level evaluation>")
	print(" |= tn=%d, fp=%d, fn=%d, tp=%d" %(tn, fp, fn, tp))
	print(" |= AUC=%.4f, Precision=%.4f, Recall=%.4f" %(auc_score, precision, recall))
	print(" |= ACC=%.4f, F1=%.4f, MCC=%.4f" %(acc, f1, mcc))



