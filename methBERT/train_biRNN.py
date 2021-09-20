#!/usr/bin/python

"""
2020/04/29 Training the models for methylation classification
"""
import argparse
from dataProcess.data_loading import *
from dataProcess.ref_util import get_fast5s
from dataProcess.visual_util import plot_curve, vis_signal_difference

from model.RNN import *

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchsummary import summary

from eval import evaluate_event_cached
from detect import detection, fix_model_state_dict, detection_cached
import time,tqdm, sys

import warnings
warnings.filterwarnings("ignore")

MIN_EPOCH=5

# cached in the memory, instead of real time loading
# train_seq, basic training
def train_seq_cached(test_region, batch_size, data_split, model, model_save_path, learn_rate, gpuID, epoch_num, use_sim=False, gen_model_path="", test_exist_model="", useSEQ=True, plotCurve=True):

	train_start = time.time()
	train_generator, test_generator, val_generator = data_split

	device = torch.device(gpuID[0] if torch.cuda.is_available() else "cpu")
	print(" |- Initial model on [%s]" %(device))

	# use the features defined in the DeepMod
	num_feat = 3
	if useSEQ: num_feat += 4

	# loading DL models
	if useSEQ:
		net = globals()[model](device, input_size=num_feat, hidden_size=100, num_layers=3).float()
	else:
		net = globals()[model](device, input_size=3).float()

	net.to(device)

	generator = None

	# the input for the loss function is the logits, the loss function itself do
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=learn_rate)

	print(" |- Total Model Parameters:", sum([p.nelement() for p in net.parameters()]))
	print(" |- Start =%s= training with [lr=%.5f]..." %(model,learn_rate))

	best_val_loss, valid_loss  =  float('inf'), float('inf')
	train_loss_record, validate_loss_record = [], []

	if test_region != None:
		print(" |- Processing data and loading into the CPU memory ... [the target region]: ", test_region)
		train_data, val_data, test_data = cached_generator_data_byRegion(test_region, train_generator, device)

		train_X, train_Y, train_align, n_train = shuffle_and_chunk_samples(batch_size, train_data, len(train_data[2]))
		val_X,   val_Y,   val_align,   n_val   = shuffle_and_chunk_samples(batch_size, val_data,   len(val_data[2]))
		test_X,  test_Y,  test_align,  n_test  = shuffle_and_chunk_samples(batch_size, test_data,  len(test_data[2]))

	else:
		X_list, Y_list, train_align_list, n_train = cached_generator_data(batch_size, train_generator, device)
		train_data = (X_list, Y_list, train_align_list)
		train_X, train_Y, train_align, n_train = shuffle_and_chunk_samples(batch_size, train_data, n_train)

		# for test/val only shuffle one time
		val_X, val_Y, val_align, n_val = cached_generator_data(batch_size, val_generator, device)
		val_X, val_Y, val_align, n_val = shuffle_and_chunk_samples(batch_size, (val_X, val_Y, val_align), n_val)

		test_X, test_Y, test_align, n_test = cached_generator_data(batch_size, test_generator, device)
		test_X, test_Y, test_align, n_test = shuffle_and_chunk_samples(batch_size, (test_X, test_Y, test_align), n_test)

	print(" |- number of chunked sample of [%d]: train-[%d], val-[%d], test-[%d]" %(batch_size, len(train_X), len(val_X), len(test_X)))

	## add the validation function of testing the loaded model for the evaluation on the testset.
	if test_exist_model != "":

		print("* Loading existed model for testing ...")
		state_dict = torch.load(test_exist_model)

		# directly loading into the existed defined model
		try:
			net.load_state_dict(state_dict)	
		except:	
			net.load_state_dict(fix_model_state_dict(state_dict))

		if len(test_Y) > 0:
			print(" \n|**** Evaluation on the test set with the given model ...")
			detection_cached(net, (test_X, test_Y, test_align, n_test), device, model, model_save_path, "both", use_sim, False)
			print("\n")
		return 0

	print(" |- Training started ...")
	for epoch in range(epoch_num):
		net.train()
		running_loss, epoch_loss = 0.0, 0.0
		skip_batch = 0

		# random shuffle samples for each iteration
		if epoch > 0:
			train_X, train_Y, train_align, n_train = shuffle_and_chunk_samples(batch_size, train_data, n_train)

		for i in range(len(train_Y)):

			#print(" [@] current GPU memeory usage in epoch -[%d] :" %(i))
			#print(torch.cuda.memory_allocated(device))
			
			seq_event, labels =train_X[i], train_Y[i]
			optimizer.zero_grad()
			outputs = net(seq_event.to(device))

			loss = criterion(outputs.to(device), labels.to(device).long())
			# this can cause the training time difference
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += float(loss.item())
			epoch_loss += float(loss.item())

			if i % 1000 == 999:
				print('[Epoch-%d,iter-%d] Train loss: %.3f' %(epoch, i+1, running_loss / 1000))
				running_loss=0.0

		epoch_loss = epoch_loss/(len(train_Y))
		train_loss_record.append(epoch_loss)

		###################################################
		# evluation on the validation set
		###################################################
		if n_val > 0:
			valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc = evaluate_event_cached(net, criterion, device, (val_X, val_Y, val_align, n_val), useSEQ, use_sim, None)
			print('* Epoch-{} | Train Loss: {:.4f} | Val Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f} | AUPRC: {:.4f}'.format(epoch, epoch_loss, valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc))
			validate_loss_record.append(valid_loss)

		# checking point
		#wait_epoch = 10
		if(valid_loss < best_val_loss and epoch >= min(MIN_EPOCH-1, epoch_num-1)):
			best_val_loss = valid_loss
			print(" |+ saving current best checkpoint at [%d]-epoch"  %(epoch))
			torch.save(net.state_dict(), model_save_path)

			if len(test_Y) > 0:
				print(" \n|**** Evaluation on the test set with the current best model ...")
				detection_cached(net, (test_X, test_Y, test_align, n_test), device, model, model_save_path, "both", use_sim, False, None)
				print("\n")

	##################### plot training curve ##################
	if plotCurve:
		print(" |+ Plot training curve ...")
		plot_curve(train_loss_record, validate_loss_record, model, model_save_path+"_curve.png")


	## saving on  the last training iteration
	if n_val == 0:
		print(" |+ saving checking point on the last epoch, as no-validation set")
		torch.save(net.state_dict(), model_save_path)

	print('Finishing training. Used [ %.1f] min.' %( (time.time()-train_start)/60))


# train_seq, basic training
def train_seq(data_split, model, model_save_path, learn_rate, gpuID, epoch_num, use_sim=False, gen_model_path="", useSEQ=True, plotCurve=True):

	train_start = time.time()
	train_generator, test_generator, val_generator = data_split

	device = torch.device(gpuID[0] if torch.cuda.is_available() else "cpu")
	print(" |- Initial model on [%s]" %(device))

	# use the features defined in the DeepMod
	num_feat = 3
	if useSEQ: num_feat += 4

	# loading DL models
	if useSEQ:
		#net = globals()[model](device, input_size=7, hidden_size=100, num_layers=3).float()
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

	#

	for epoch in range(epoch_num):
		net.train()
		running_loss, epoch_loss = 0.0, 0.0
		skip_batch = 0

		for i, data in enumerate(train_generator, 0):
			inputs, labels = data

			# generator data processing part
			# batch data loader this part can be put into the function
			if len(labels) == 0: 
				skip_batch += 1
				continue

			if useSEQ:
				seq = F.one_hot(inputs[0]) #.permute(0,2,1)
				seq_event = torch.cat((seq.float().to(device), inputs[2].to(device)) , -1)

				if use_sim:
					#seq_event = torch.cat((seq.float(), inputs[2][:,:,0:2]-inputs[3][:,:,0:2]), -1)
					seq_event = torch.cat((seq.float().to(device), inputs[2].to(device), inputs[2].to(device)-inputs[3].to(device)), -1)

				if gen_model_path != "":
					fake_signal = generator(seq.to(device).float())
					seq_event = torch.cat((seq.float().to(device), inputs[2].to(device), fake_signal.to(device)), -1)
			else:
				seq_event = inputs[2]
			#######################
			
			optimizer.zero_grad()
			outputs = net(seq_event.to(device))

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
		if len(val_generator) > 0:
			valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc = evaluate_event(net, criterion, device, val_generator, useSEQ, use_sim, generator)
			print('* Epoch-{} | Train Loss: {:.4f} | Val Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f} | AUPRC: {:.4f}'.format(epoch, epoch_loss, valid_loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc))
			validate_loss_record.append(valid_loss)

		# checking point
		#wait_epoch = 10 
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

	## saving on the last training iteration for no validation training
	if len(val_generator) == 0:
		print(" |+ saving checking point on the last epoch, as no-validation set")
		torch.save(net.state_dict(), model_save_path)

	print('Finishing training. Used [ %.1f] min.' %( (time.time()-train_start)/60))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='<PyTorch DeepMehtylation Training>')

	parser.add_argument('--model',     default='CNN',      type=str, required=True,  help="DL models used for the training.")
	parser.add_argument('--model_dir', action="store",     type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--gpu',       default="cuda:0",   type=str, required=False, help='GPU Device(s) used for training')
	parser.add_argument('--epoch',     default=50,         type=int, required=False, help='Training epcohs')
	parser.add_argument('--batch_size',default=128,         type=int,  required=False, help="batch_size of the training")
	parser.add_argument('--read_loading_batch_size',default=100,       type=int,  required=False, help="read batch size for loading the data")
	parser.add_argument('--dataset',   default="",         type=str, required=False,  help='dataset name')
	parser.add_argument('--dataset_extra', default="",     type=str, required=False, help='Additional data tag')
	
	parser.add_argument('--positive_control_dataPath',   default="", type=str, required=True,  help='positive control fast5 dataPath')
	parser.add_argument('--negative_control_dataPath',   default="", type=str, required=True, help='negative control fast5 dataPath')

	parser.add_argument('--motif',     default="CG",       type=str, required=True,  help='motif lists, currently one Motif type only')
	parser.add_argument('--m_shift',   default=0,          type=int, required=False, help='Methylation target local start position')
	parser.add_argument('--w_len',     default=21,         type=int, required=False, help='input nucleotide window length')
	parser.add_argument('--unseg',     action="store_true",          required=False, help='option for un-event segment [default:False]')
	parser.add_argument('--use_sim',   action="store_true",          required=False, help='use the simulation for input normalization [default:False]')
	parser.add_argument('--gen_model_path',   default="" ,                  required=False, help='use the generator for additional feature in the given model path [default:""]')
	parser.add_argument('--group_eval',action="store_true",          required=False, help='use the simulation for input normalization [default:False]')
	parser.add_argument('--num_worker',default=-1,          type=int, required=False, help='number of working for loading the data')
	parser.add_argument('--lr',		   default=1e-3,      type=float, required=False, help='learning rate for the training')

	parser.add_argument('--lm',        default="",        type=str,  required=False, help="Loading the previous trained model for the testing.")
	parser.add_argument('--vis_data',  default="",        type=str,  required=False, help="viualization signals")
	
	parser.add_argument('--test_region',default = [],     nargs="+",  required=False, help="prepare samples in the test region for testing")
	parser.add_argument('--train_test_split_ratio', type=float, default=[0.8, 0.1], nargs="+",required=False, help="prepare samples according to this ratio")
	parser.add_argument('--data_balance_adjust', action="store_true", required=False, help='Data balance adjustment [default:False]')
	
	parser.add_argument('--test_exist_model', default="",  type=str, required=False,  help="Test the model performance with existed model.")


	args = parser.parse_args()
	cores = mp.cpu_count()
	if args.num_worker > 0 and args.num_worker < cores:
		cores = args.num_worker

	gpu_list = [item for item in args.gpu.split(',')]
	motif = [item.upper() for item in args.motif.split(',')]

	print("\n[+] Methylation %s-motif Model Training for %d-th position [%s] of nanopore fast5 data ..." %("".join(motif), args.m_shift, motif[0][args.m_shift]))
	
	# add direction path information
	if args.dataset == "" and args.dataset_extra =="":

		if args.positive_control_dataPath == "" or args.negative_control_dataPath == "":
			print("[Data Error]: Please cheching the positive/negative fast5 data path!")
			exit()

		meth_fold_path = args.positive_control_dataPath
		pcr_fold_path  = args.negative_control_dataPath


	if args.data_balance_adjust:
		print(" [***] Read sample balance adjustment is [ON]!!")

	print(" [*] Data caching read-batch size is [%d]" %(args.read_loading_batch_size))
	print(" [*] Training sample-batch size is [%d]" %(args.batch_size))
	
	# region-based split
	target_test_region = None
	if len(args.test_region) == 3:
		#test_region = ('NC_000913.3', 1000000,2000000)
		target_test_region = (args.test_region[0], int(args.test_region[1]), int(args.test_region[2]))
		print(" |-* Split data by regions and test-data is prepared in ", target_test_region)
		data_split = load_from_2folds_select_testRegion(target_test_region, meth_fold_path, pcr_fold_path, cores, args.read_loading_batch_size, motif, args.m_shift, args.w_len, args.data_balance_adjust)
	else:
		print(" |-* Split data by reads according to the proporation [Train=%f, Test=%f]" %(args.train_test_split_ratio[0], args.train_test_split_ratio[1]))
		data_split = load_from_2folds(meth_fold_path, pcr_fold_path, cores, args.read_loading_batch_size, motif, args.m_shift, args.w_len, args.train_test_split_ratio, args.data_balance_adjust)

	# data visualization 
	if args.vis_data != "":
		print(" |- Visualization train signals ...")
		vis_signal_difference(data_split[0], args.vis_data + "/" + args.dataset_extra + ".png")
		#sys.exit(0)

	print(" |- using [%d] cores." %(cores))

	if args.use_sim:
		print(" |- *using simulation squiggle in the input!" )

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

	train_seq_cached(target_test_region, args.batch_size, data_split, args.model, args.model_dir, args.lr, gpu_list, args.epoch, args.use_sim, args.gen_model_path, args.test_exist_model)
	




