#!/usr/bin/python
"""
2020/05/08 Training the models for methylation classification
"""
import argparse, time
from dataProcess.data_loading import *
from dataProcess.ref_util import get_fast5s
from dataProcess.visual_util import plot_curve

from model.BERT import *
from model.BERT_plus import *

from eval import evaluate_bert, evaluate_signal, evaluate_hierbert
from detect import detection, fix_model_state_dict

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

MIN_EPOCH=5

def train_bert_plus(data_split, model, model_save_path, learn_rate, gpuID, epoch_num, use_sim=False, m_shift=0, m_len=2, w_len=21, useSEQ=True, plotCurve=True):
	
	train_start = time.time()
	train_generator, test_generator, val_generator = data_split

	device = torch.device(gpuID[0] if torch.cuda.is_available() else "cpu")
	print(" |- Initial model on [%s]" %(device))
	#print(" |- Motif [shift=%d, len=%d]" %(m_shift, m_len))

	# loading DL models
	bert = BERT_plus(vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0, motif_shift=m_shift, motif_len=m_len, seq_len=w_len).float()

	bert.to(device)

	if len(gpuID) > 1 and torch.cuda.device_count() > 1:
		print(" |- Using %d/%d GPUs for training: %s" %(len(gpuID), torch.cuda.device_count(), ",".join(gpuID)))
		bert = nn.DataParallel(bert, device_ids=gpuID)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(bert.parameters(), lr=learn_rate)


	print(" |- Total Model Parameters:", sum([p.nelement() for p in bert.parameters()]))
	print(" |- Start =BERT_plus(relative poisitional embedding)= training with [lr=%.5f]..." %(learn_rate))

	best_val_loss = float('inf')
	train_loss_record, validate_loss_record = [], []

	## confirm whether the parameters are in the lists
	show_param_space = False
	if show_param_space:
		pn = []
		for name, param in bert.named_parameters():
			if param.requires_grad:
				pn.append(name)	
			print(pn[:10])

	for epoch in range(epoch_num):
		bert.train()
		running_loss, epoch_loss = 0.0, 0.0

		for i, data in enumerate(train_generator, 0):
			inputs, labels = data
			if len(labels) == 0: continue

			# merging the seq with events information
			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float(), inputs[2]) , -1)
			else:
				# single infomrationonly
				seq_event = inputs[2]

			optimizer.zero_grad()
			outputs = bert(seq_event.to(device), None)

			#loss = criterion(outputs, labels.to(device).unsqueeze(1).type_as(outputs))	
			loss = criterion(outputs, labels.to(device).long())

			loss.backward(retain_graph=True)
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			epoch_loss += loss.item()
			if i % 1000 == 999:
				print('[Epoch-%d,iter-%d] Train loss: %.3f' %(epoch, i+1, running_loss / 1000))
				running_loss=0.0

		epoch_loss = epoch_loss/len(train_generator)
		train_loss_record.append(epoch_loss)

		###################################################
		# evluation on the validation set
		###################################################
		## HPC stops in the evluate_bert function
		valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity = evaluate_bert(bert, criterion, device, val_generator, useSEQ, use_sim)
		
		print('Epoch-{} | Train Loss: {:.4f} | Val Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f}'.format(epoch, epoch_loss, valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity))
		validate_loss_record.append(valid_loss)

		if(valid_loss < best_val_loss and epoch >= min(MIN_EPOCH-1, epoch_num-1)):
			best_val_loss = valid_loss
			print(" |----> saving current best checkpoint at [%d]-epoch"  %(epoch))
			if len(gpuID) > 1:
				torch.save(bert.module.state_dict(), model_save_path)
			else:
				torch.save(bert.state_dict(), model_save_path)

			if len(test_generator) > 0:
				print(" \n|+ **** Evaluation with the current best model ***")
				detection(bert, test_generator, device, model, model_save_path, "both", use_sim, False)
				print("\n")
				#valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity = evaluate_bert(bert, criterion, device, test_generator, useSEQ, use_sim)

	print('Finishing training. Used [ %.1f] min.' %( (time.time()-train_start)/60))

	##################### plot training curve ##################
	if plotCurve:
		plot_curve(train_loss_record, validate_loss_record, model, model_save_path+"_curve.png")


def train_bert_basic(data_split, model, model_save_path, learn_rate, gpuID, epoch_num, use_sim=False, useSEQ=True, plotCurve=True):

	train_start = time.time()

	train_generator, test_generator, val_generator = data_split

	device = torch.device(gpuID[0] if torch.cuda.is_available() else "cpu")
	print(" |- Initial model on [%s]" %(device))

	# loading DL models
	# previous basic model
	bert = BERT(vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0).float()

	bert.to(device)
	if len(gpuID) > 1 and torch.cuda.device_count() > 1:
		print(" |- Using %d/%d GPUs for training: %s" %(len(gpuID), torch.cuda.device_count(), ",".join(gpuID)))
		bert = nn.DataParallel(bert, device_ids=gpuID)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(bert.parameters(), lr=learn_rate)

	print(" |- Total Model Parameters:", sum([p.nelement() for p in bert.parameters()]))
	print(" |- Start =BERT_basic= training with [lr=%.5f]..." %(learn_rate))


	best_val_loss = float('inf')
	train_loss_record, validate_loss_record = [], []

	for epoch in range(epoch_num):
		bert.train()
		running_loss, epoch_loss = 0.0, 0.0
		skip_batch = 0

		for i, data in enumerate(train_generator, 0):
			inputs, labels = data

			if len(labels) == 0: 
				skip_batch += 1
				continue

			# merging the seq with events information
			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				if use_sim:
					seq_event = torch.cat((seq.float(), inputs[2]-inputs[3]), -1)
				else:
					seq_event = torch.cat((seq.float(), inputs[2]) , -1)
			else:
				# single infomrationonly
				seq_event = inputs[2]

			optimizer.zero_grad()

			outputs = bert(seq_event.to(device), None)
			loss = criterion(outputs, labels.to(device).long())

			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			epoch_loss += loss.item()
			if i % 1000 == 999:
				print('[Epoch-%d,iter-%d] Train loss: %.3f' %(epoch, i+1, running_loss / 1000))
				running_loss=0.0

		epoch_loss = epoch_loss/(len(train_generator)-skip_batch)
		train_loss_record.append(epoch_loss)

		###################################################
		# evluation on the validation set
		###################################################

		valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity = evaluate_bert(bert, criterion, device, val_generator, useSEQ, use_sim)
		print('Epoch-{} | Train Loss: {:.4f} | Val Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f}'.format(epoch, epoch_loss, valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity))
		validate_loss_record.append(valid_loss)

		# checking point
		if(valid_loss < best_val_loss and epoch >= min(MIN_EPOCH-1, epoch_num-1)):
			best_val_loss = valid_loss
			print(" |+ saving current best checkpoint at [%d]-epoch"  %(epoch))
			torch.save(bert.state_dict(), model_save_path)

			if len(test_generator) > 0:
				print(" \n|+ **** Evaluation with the current best model ***")
				detection(bert, test_generator, device, model, model_save_path, "both", use_sim, False)
				print("\n")

	##################### plot training curve ##################
	if plotCurve:
		plot_curve(train_loss_record, validate_loss_record, model, model_save_path+"_curve.png")
	
	print('Finishing training. Used [ %.1f] min.' %( (time.time()-train_start)/60))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='<PyTorch DeepMehtylation Training>')

	parser.add_argument('--model',     default='BERT',   type=str, required=True,  help="DL models used for the training.")
	parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--gpu',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
	parser.add_argument('--epoch',     default=50,       type=int, required=False, help='Training epcohs')
	parser.add_argument('--dataset',   default="",       type=str, required=True, help='dataset name')
	parser.add_argument('--dataset_extra', default="",     type=str, required=False, help='Additional data tag')
	
	parser.add_argument('--positive_control_dataPath',   default="", type=str, required=False,  help='positive control fast5 dataPath')
	parser.add_argument('--negative_control_dataPath',   default="", type=str, required=False, help='negative control fast5 dataPath')

	parser.add_argument('--motif',     default="CG",     type=str, required=True, help='motif lists, currently one Motif type only')
	parser.add_argument('--m_shift',    default=0,        type=int, required=True, help='Methylation target local start position')
	parser.add_argument('--w_len',     default=21,       type=int, required=False, help='input nucleotide window length')
	parser.add_argument('--num_worker',default=-1,          type=int, required=False, help='number of working for loading the data')
	parser.add_argument('--lr',		   default=1e-3,      type=float, required=False, help='learning rate for the training')

	parser.add_argument('--lm',        default="",        type=str,  required=False, help="Loading the previous trained model for the testing.")

	parser.add_argument('--unseg',     action="store_true",          required=False, help='option for un-event segment [default:False]')
	parser.add_argument('--use_sim',   action="store_true",          required=False, help='use the simulation for input normalization [default:False]')
	parser.add_argument('--group_eval',action="store_true",          required=False, help='use the simulation for input normalization [default:False]')
	parser.add_argument('--vis_data',   default="",        type=str,required=False, help='viualization signals')


	args = parser.parse_args()
	cores = mp.cpu_count()
	if args.num_worker > 0 and args.num_worker < cores:
		cores = args.num_worker
		
	gpu_list = [item for item in args.gpu.split(',')]
	motif = [item.upper() for item in args.motif.split(',')]

	print("\n[+] Methylation %s-motif Model Training for %d-th position [%s] of nanopore fast5 data ..." %("".join(motif), args.m_shift, motif[0][args.m_shift]))

	home_path="/nanopore"
	# loading from dine
	if args.dataset == "simpson_ecoli":
		meth_fold_path = home_path + "/data/dev/ecoli_er2925.pcr_MSssI.r9.timp.061716.fast5/pass"
		pcr_fold_path  = home_path +  "/data/dev/ecoli_er2925.pcr.r9.timp.061716.fast5/pass"

	if args.dataset == "simpson_human":
		meth_fold_path = home_path + "/data/dev/NA12878.pcr_MSssI.r9.timp.081016.fast5/pass"
		pcr_fold_path  = home_path + "/data/dev/NA12878.pcr.r9.timp.081016.fast5/pass"	

	if args.dataset == "stoiber_ecoli":
		pcr_fold_path  = home_path + "/data/Stoiber/5mC/Control"

		if args.dataset_extra == "M_Hhal_gCgc":
			meth_fold_path = home_path + "/data/Stoiber/5mC/M_Hhal_gCgc"
		elif args.dataset_extra == "M_Mpel_Cg":
			meth_fold_path = home_path + "/data/Stoiber/5mC/M_Mpel_Cg"
		elif args.dataset_extra == "M_Sssl_Cg":
			meth_fold_path = home_path + "/data/Stoiber/5mC/M_Sssl_Cg"

		# add the 6mA data loading part
		elif args.dataset_extra == "M_EcoRI_gaAttc":
			meth_fold_path = home_path + "/data/Stoiber/6mA/M_EcoRI_gaAttc"
		elif args.dataset_extra == "M_TaqI_tcgA":
			meth_fold_path = home_path + "/data/Stoiber/6mA/M_TaqI_tcgA"
		elif args.dataset_extra == "M_dam_gAtc":
			meth_fold_path = home_path + "/data/Stoiber/6mA/M_dam_gAtc"
		elif args.dataset != "" and dataset_extra !="":
			print("[Error!] Please assign the corrected methyltation data name for the stoiber data-set")

	# add direction path information
	if args.dataset == "" and dataset_extra =="":

		if args.positive_control_dataPath == "" or args.negative_control_dataPath == "":
			print("[Data Error]: Please cheching the positive/negative fast5 data path!")
			exit()

		meth_fold_path = args.positive_control_dataPath
		pcr_fold_path  = args.negative_control_dataPath

	train_test_split = (0.8, 0.1)

	data_split = load_from_2folds(meth_fold_path, pcr_fold_path, cores, 10, motif, args.m_shift, args.w_len, train_test_split, False)

	# data visualization 
	if args.vis_data != "":
		print(" |- Visualization train signals ...")
		vis_signal_difference(data_split[0], args.vis_data + "/" + args.dataset_extra + ".png")

	print(" |- using [%d] cores." %(cores))
	if args.use_sim:
		print(" |- *using simulation squiggle in the input!" )

	if args.lm != "":
		print(" |- testing the test set using the previous loaded model ...")
		train_generator, test_generator, val_generator = data_split
		device = torch.device(gpu_list[0] if torch.cuda.is_available() else "cpu")

		print("* Loading [%s] model ..." %(args.model))
		state_dict = torch.load(args.lm)
		
		net = globals()[args.model](vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0).float()

		try:
			net.load_state_dict(state_dict)	
		except:	
			net.load_state_dict(fix_model_state_dict(state_dict))
		
		print(" \n|**** Evaluation on the test set with the current best model ...")
		detection(net, test_generator, device, args.model, args.lm, "both", args.use_sim, False, None)
		print("\n")
		exit()

	if args.model == "BERT":
		train_bert_basic(data_split, args.model, args.model_dir, args.lr, gpu_list, args.epoch, args.use_sim)

	if args.model == "BERT_plus":
		train_bert_plus(data_split, args.model, args.model_dir, args.lr, gpu_list, args.epoch, args.use_sim, args.m_shift, len(args.motif), args.w_len)



	


