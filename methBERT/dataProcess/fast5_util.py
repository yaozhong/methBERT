#!/usr/bin/python

import h5py

from .visual_util import *
from .ref_util import _get_readid_from_fast5, \
     _get_alignment_attrs_of_each_strand

from statsmodels import robust
import numpy as np
import collections


""" get the raw signal and event from fast5 """
# this module is based on DeepSignal
def get_signal_event_from_fast5(file_name, reads_group="Raw/Reads", 
	correct_group="RawGenomeCorrected_001", correct_subgroup="BaseCalled_template"):

	try:
		fast5_data = h5py.File(file_name, 'r')
	except IOError:
		raise IOError("Loading fast5 error! Please check fast5 file!")

	# get signal data
	try:
		raw = list(fast5_data[reads_group].values())[0]
		signal =  raw['Signal'][()]
	except Exception:
		raise RuntimeError('Raw data is not stored in Raw/Reads/Read_[read#]')
		#return None, None, None

	# scaling parameters
	global_key = "UniqueGlobalKey/"
	try:
		channel_info = dict(list(fast5_data[global_key + 'channel_id'].attrs.items()))
		digi = channel_info['digitisation']
		parange = channel_info['range']
		offset = channel_info['offset']
		scaling = parange / digi
	except IOError:
		print("Scaling parameters extraction fail!")
		return None, None, None

	## rescale
	signal = np.array(scaling * (signal + offset), dtype=np.float)
	## normalization
	sshift, sscale = np.median(signal), np.float(robust.mad(signal))
	norm_signal = (signal - sshift) / sscale

	# get basecall information
	"""  not used in the intial development stage
	try:
		basecall = fast5_data['/Analyses/Basecall_1D_000/'+ correct_subgroup + '/Fastq'] 
		basecall = basecall[()].split("\n")
	except Exception:
		raise RuntimeError('Fail to extract Fastq information')
	"""

	# get events
	try:
		event = fast5_data['/Analyses/'+ correct_group + '/' + correct_subgroup + '/Events']
		corr_attrs = dict(list(event.attrs.items()))
	except Exception:
		raise RuntimeError('Events not found. Use Guppy + tombo for generate corrected events information.')
		#return None, None, None

	read_start_position = corr_attrs['read_start_rel_to_raw']

	starts = list(map(lambda x: x+read_start_position, event['start']))
	lengths = event['length'].astype(np.int)
	base = [x.decode("UTF-8") for x in event['base']]
	assert(len(starts) == len(lengths) and len(lengths) == len(base))
	events = list(zip(starts, lengths, base))

	# get alignment information
	try:
		corrgroup_path = '/'.join(['Analyses', correct_group])
		if '/'.join([corrgroup_path, correct_subgroup, 'Alignment']) in fast5_data:
			# original used alignment information processing
			readname = _get_readid_from_fast5(fast5_data)
			strand, alignstrand, chrom, chrom_start = _get_alignment_attrs_of_each_strand('/'.join([corrgroup_path, correct_subgroup]),fast5_data)
			

		align_info = (chrom, chrom_start, strand, alignstrand, readname)

	except IOError:
		print("Alignment infomration exatraction fail.")
		return None, None, None

	norm_signal = norm_signal.astype(np.float32)

	#if len(events) < 500:
		# print("	>> event length is less than 500 filtered ... " , len(events))
	#	return None, None, None

	return norm_signal, events, align_info



if __name__ == "__main__":

	file_name = "../../data/samples/LomanLabz_PC_20161125_FNFAF01132_MN17024_sequencing_run_20161124_Human_Qiagen_1D_R9_4_96293_ch389_read2521_strand.fast5"
	
	## extraction features
	read_motif_input = single_read_input_extract(file_name)

	print(read_motif_input[0])




	