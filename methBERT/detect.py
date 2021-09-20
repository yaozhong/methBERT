#!/usr/bin/python

"""
2020/05/08 Training the models for methylation classification
"""
import argparse
from dataProcess.data_loading import *
from dataProcess.ref_util import get_fast5s

from model.BERT import *
from model.BERT_plus import BERT_plus
from model.RNN import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from eval import evaluate_signal, evaluate_event, evaluate_single, evaluate_unseg_seq, evaluate_print, evaluate_event_cached
from collections import OrderedDict
from torchsummary import summary

## correct the multi-gpu model saving issue ##
def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

# this function is used in the training
def detection(net, data_gen, device, model, model_save_path, eval_mode, use_sim=False, unseg=False, generator=None, permutate=False):

	net.to(device)		
	criterion = nn.CrossEntropyLoss()

	print("* Methylation detection for nanopore reads ...")
	if eval_mode == "both":
		if unseg == True:
			loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc = evaluate_unseg_seq(net, criterion, device, data_gen, True)
		else:
			loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc = evaluate_event(net, criterion, device, data_gen, True, use_sim, generator, permutate)
		
		print('Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f} | AUPRC: {:.4f}'.format(loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc))
	
	elif eval_mode == "motif_only":
		loss, accuracy = evaluate_single(net, criterion, device, data_gen, True)
		print('Loss: {:.4f} | Accuracy: {:.4f}'.format(loss, accuracy))


def detection_cached(net, data_gen, device, model, model_save_path, eval_mode, use_sim=False, unseg=False, generator=None, permutate=False):

	net.to(device)		
	criterion = nn.CrossEntropyLoss()

	print("* Methylation detection for nanopore reads ...")
	if eval_mode == "both":
		if unseg == True:
			loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc = evaluate_unseg_seq(net, criterion, device, data_gen, True)
		else:
			loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc = evaluate_event_cached(net, criterion, device, data_gen, True, use_sim, generator, permutate)

		print('Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f} | AUPRC: {:.4f}'.format(loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc))
	
	# not revise for the cached running
	elif eval_mode == "motif_only":
		loss, accuracy = evaluate_single(net, criterion, device, data_gen, True)
		print('Loss: {:.4f} | Accuracy: {:.4f}'.format(loss, accuracy))	



# this function is used in the independnent evaluation, which loads the model.
def detection_run(data_gen, device, model, model_save_path, eval_mode, use_sim=False, ref_genome=None, output_file="../output.tsv", unseg=False, generator=None):

	print("* Loading trained model ...")
	state_dict = torch.load(model_save_path)

	if model == "biRNN_basic":
		net = globals()[model](device).float()
	elif model == "BERT":
		# BERT model parameter are fixed
		net = globals()[model](vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0).float()
	elif model == "BERT_plus":
		# parameters here are also fixed for the CG evaluation
		net = BERT_plus(vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0, motif_shift=0, motif_len=2).float()
		#net = nn.DataParallel(net)

	net.to(device)

	try:
		net.load_state_dict(state_dict)	
	except:	
		net.load_state_dict(fix_model_state_dict(state_dict))
		
	criterion = nn.CrossEntropyLoss()

	#for name, param in net.state_dict().items():
	#	print(name, param.data)

	print("* Methylation detection for nanopore reads ...")
	if eval_mode == "both":
		if unseg == True:
			loss, accuracy, auc_score, precision, recall, sensitivity, specificity = evaluate_unseg_seq(net, criterion, device, data_gen, True)
		else:
			loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc = evaluate_event(net, criterion, device, data_gen, True, use_sim, generator)

		print('Loss: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}| Precision: {:.4f}, Recall: {:.4f} | Sensitivity: {:.4f}, Specificity: {:.4f}, AUPRC: {:.4f}'.format(loss, accuracy, auc_score, precision, recall, sensitivity, specificity, auprc))
	
	elif eval_mode == "motif_only":
		loss, accuracy = evaluate_single(net, criterion, device, data_gen, True)
		print('Loss: {:.4f} | Accuracy: {:.4f}'.format(loss, accuracy))

	elif eval_mode == "test_mode":
		# output the same results as the deepSignal
		evaluate_print(net, criterion, device, data_gen, True, ref_genome, output_file)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='<PyTorch DeepMehtylation Training>')

	parser.add_argument('--model',     default='biRNN_basic',    type=str, required=True,  help="DL models used for the training.")
	parser.add_argument('--model_dir', action="store",     type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--gpu',       default="cuda:0",   type=str, required=False, help='GPU Device(s) used for training')
	parser.add_argument('--dataset',   default="",         type=str, required=False,  help='dataset name')
	parser.add_argument('--dataset_extra', default="",     type=str, required=False, help='Additional data tag')

	parser.add_argument('--motif',     default="CG",       type=str, required=True,  help='motif lists, currently one Motif type only')
	parser.add_argument('--m_shift',   default=0,          type=int, required=False, help='Methylation target local start position')
	parser.add_argument('--evalMode',  default="test_mode",     type=str, required=False, help='Evluation mode, [test_mode/motif_only, both]')
	parser.add_argument('--w_len',     default=21,         type=int, required=False, help='input nucleotide window length')
	parser.add_argument('--unseg',     default=False,      type=bool,required=False, help='option for un-event segment [default:False]')
	parser.add_argument('--num_worker',default=-1,          type=int, required=False, help='number of working for loading the data')
	parser.add_argument('--ref_genome', action="store",     type=str, required=True,  help="Reference genome. used for the strand position calcuation as in deepsignal")
	parser.add_argument('--output_file',action="store",     type=str, required=True,  help="output file of the prediction results")
	parser.add_argument('--fast5_fold',default="", action="store",      type=str, required=True,  help="target fast5 files for methylation analysis")

	#random_seed = 123
	#torch.manual_seed(random_seed)

	args = parser.parse_args()
	cores = mp.cpu_count()
	if args.num_worker > 0 and args.num_worker < cores:
		cores = args.num_worker

	gpu_list = [item for item in args.gpu.split(',')]
	motif = [item.upper() for item in args.motif.split(',')]

	print("[+] Detecting methylation %s-motif with for %d-th position [%s] for nanopore fast5 data ..." %("".join(motif), args.m_shift, motif[0][args.m_shift]))

	if args.fast5_fold != "":
		meth_fold_path = args.fast5_fold
		print(meth_fold_path)
			
	# evluation mode
	if args.evalMode == "both":
		data_gen, _, _ = load_from_2folds(meth_fold_path, pcr_fold_path, cores, 20, motif, args.m_shift, args.w_len, (1,0))
		
	elif args.evalMode == "motif_only" or args.evalMode == "test_mode":
		data_gen = load_from_single_folds(meth_fold_path, cores, 1, 10, motif, args.m_shift, args.w_len)

	device = torch.device(gpu_list[0] if torch.cuda.is_available() else "cpu")

	# detection evluation
	if args.unseg:
		detection_run(data_gen, device, args.model, args.model_dir, args.evalMode, True, None)
	else:
		detection_run(data_gen, device, args.model, args.model_dir, args.evalMode, False, args.ref_genome, args.output_file)

