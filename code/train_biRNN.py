#!/usr/bin/python

"""
2020/04/29 Training the models for methylation classification
"""

import argparse
from dataProcess.data_loading import *
from dataProcess.ref_util import get_fast5s
from dataProcess.visual_util import plot_curve, vis_signal_difference

from model.simple_CNN import *
from model.RNN import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from eval import evaluate_signal, evaluate_event, evaluate_unseg_seq, evaluate_event_for_GANsim
from detect import detection, fix_model_state_dict
import time,tqdm, sys

MIN_EPOCH=5

# train_seq, basic training
def train_seq(data_split, model, model_save_path, learn_rate, gpuID, epoch_num, use_sim=False, gen_model_path="", useSEQ=True, plotCurve=True):

	train_start = time.time()
	train_generator, test_generator, val_generator = data_split

	device = torch.device(gpuID[0] if torch.cuda.is_available() else "cpu")
	print(" |- Initial model on [%s]" %(device))

	num_feat = 3
	if useSEQ: num_feat += 4

	# loading DL models
	if useSEQ:
		net = globals()[model](device, input_size=num_feat, hidden_size=100, num_layers=3).float()
	else:
		net = globals()[model](device, input_size=3).float()

	net.to(device)

	generator = None
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=learn_rate)

	print(" |- Total Model Parameters:", sum([p.nelement() for p in net.parameters()]))

	# training iterations
	print(" |- Start =%s= training with [lr=%.5f]..." %(model,learn_rate))

	best_val_loss, valid_loss  =  float('inf'), float('inf')
	train_loss_record, validate_loss_record = [], []

	for epoch in range(epoch_num):

		net.train()
		running_loss, epoch_loss = 0.0, 0.0
		skip_batch = 0

		for i, data in enumerate(train_generator, 0):
			inputs, labels = data

			# generator data processing part
			# batch data loader.
			if len(labels) == 0: 
				skip_batch += 1
				continue

			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float().to(device), inputs[2].to(device)) , -1)

				if use_sim:
					seq_event = torch.cat((seq.float().to(device), inputs[2].to(device), inputs[2].to(device)-inputs[3].to(device)), -1)

			else:
				seq_event = inputs[2]
			
			optimizer.zero_grad()
			outputs = net(seq_event.to(device))

			loss = criterion(outputs, labels.to(device).long())
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			epoch_loss += loss.item()

			# print statistics
			if i % 1000 == 999:
				print('[Epoch-%d,iter-%d] Train loss: %.3f' %(epoch, i+1, running_loss / 1000))
				running_loss=0.0

		epoch_loss = epoch_loss/(len(train_generator)-skip_batch)
		train_loss_record.append(epoch_loss)

		###################################################
		# evluation on the validation set
		###################################################
		if len(val_generator) > 0:
			valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity = evaluate_event(net, criterion, device, val_generator, useSEQ, use_sim, generator)
			print('* Epoch-{} | Train Loss: {:.4f} | Val Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f}'.format(epoch, epoch_loss, valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity))
			validate_loss_record.append(valid_loss)

		# Update and save the current best loss parameters
		if(valid_loss < best_val_loss and epoch >= min(MIN_EPOCH-1, epoch_num-1)):
			best_val_loss = valid_loss
			print(" |+ saving current best checkpoint at [%d]-epoch"  %(epoch))
			torch.save(net.state_dict(), model_save_path)

			if len(test_generator) > 0:
				print(" \n|**** Evaluation on the test set with the current best model ...")
				detection(net, test_generator, device, model, model_save_path, "both", use_sim, False, generator)
				print("\n")

	##################### plot training curve ##################
	if plotCurve:
		print(" |+ Plot training curve ...")
		plot_curve(train_loss_record, validate_loss_record, model, model_save_path+"_curve.png")

	## saving on the last training iteration
	if len(val_generator) == 0:
		print(" |+ saving checking point on the last epoch, as no-validation set")
		torch.save(net.state_dict(), model_save_path)

	print('Finishing training. Used [ %.1f] min.' %( (time.time()-train_start)/60))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='<PyTorch DeepMehtylation Training>')

	parser.add_argument('--model',     default='CNN',      type=str, required=True,  help="DL models used for the training.")
	parser.add_argument('--model_dir', action="store",     type=str, required=True,  help="Directory for saving the trained model.")
	parser.add_argument('--gpu',       default="cuda:0",   type=str, required=False, help='GPU Device(s) used for training')
	parser.add_argument('--epoch',     default=50,         type=int, required=False, help='Training epcohs')
	parser.add_argument('--dataset',   default="",         type=str, required=False,  help='Dataset name')
	parser.add_argument('--dataset_extra', default="",     type=str, required=False, help='used as the addtitional for Stoiber/Simpson dataset')

	parser.add_argument('--positive_control_dataPath',   default="", type=str, required=False,  help='positive control fast5 dataPath')
	parser.add_argument('--negative_control_dataPath',   default="", type=str, required=False, help='negative control fast5 dataPath')

	parser.add_argument('--motif',     default="CG",       type=str, required=True,  help='motif lists, currently one Motif type only')
	parser.add_argument('--m_shift',   default=-1,          type=int, required=False, help='Methylation target local start position')
	parser.add_argument('--w_len',     default=21,         type=int, required=False, help='Input nucleotide window length')

	# evaluation options
	parser.add_argument('--group_eval',action="store_true", required=False, help='use the simulation for input normalization [default:False]')
	parser.add_argument('--num_worker',default=1,   type=int, required=False, help='number of working for loading the data')
	parser.add_argument('--lr',		   default=1e-4, type=float, required=False, help='learning rate for the training')

	parser.add_argument('--lm',        default="",        type=str,  required=False, help="Loading the previous trained model for the testing.")
	parser.add_argument('--vis_data',  default="",        type=str,  required=False, help="viualization signals")

	# the following options is under develpment
	parser.add_argument('--unseg',     action="store_true", required=False, help='option for un-event segment [default:False]')
	parser.add_argument('--use_sim',   action="store_true", required=False, help='use the simulation for input normalization [default:False]')
	parser.add_argument('--gen_model_path',   default="" ,  required=False, help='use the generator for additional feature in the given model path [default:""]')
	
	args = parser.parse_args()
	cores = mp.cpu_count()
	if args.num_worker > 0 and args.num_worker < cores:
		cores = args.num_worker

	gpu_list = [item for item in args.gpu.split(',')]
	motif = [item.upper() for item in args.motif.split(',')]

	print("\n[+] Methylation %s-motif Model Training for %d-th position [%s] of nanopore fast5 data ..." %("".join(motif), args.m_shift, motif[0][args.m_shift]))
	print("	 - This may take time for processing fast5 files and generating features (depending on input) ")
	
	# add direction path information
	if args.dataset == "" and args.dataset_extra =="":
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
		print(" |- using simulation squiggle in the input!" )

	if args.lm != "":
		print(" |- testing the test set using the previous loaded model ...")
		train_generator, test_generator, val_generator = data_split
		device = torch.device(gpu_list[0] if torch.cuda.is_available() else "cpu")

		print("* Loading model ...")
		state_dict = torch.load(args.lm)
		
		net = globals()[args.model](device).float()
		
		try:
			net.load_state_dict(state_dict)	
		except:	
			net.load_state_dict(fix_model_state_dict(state_dict))
		
		print(" \n|**** Evaluation on the test set with the current best model ...")
		detection(net, test_generator, device, args.model, args.lm, "both", args.use_sim, False, None)
		print("\n")
		exit()

	train_seq(data_split, args.model, args.model_dir, args.lr, gpu_list, args.epoch, args.use_sim, args.gen_model_path)





