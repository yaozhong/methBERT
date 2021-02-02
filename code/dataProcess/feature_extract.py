""" features used for methylation prediction in DeepSignal """

# search for the motifs sequences in a given sequence.
import numpy as np
from dataProcess.ref_util import _get_central_signals, _get_alignment_info_from_fast5
from dataProcess.fast5_util import get_signal_event_from_fast5

# note the centric shift is also important
def get_methysite_in_ref(seqstr, motifs, methy_shift=1, singleton=False):

	# remove duplicate
    motifs = list(set(motifs))
    ref_seq_len = len(seqstr)
    # assume all motifs has the same length, only test CG
    motiflen = len(motifs[0])
    num_non_singleton = 0

    sites = []
    for i in range(0, ref_seq_len - motiflen + 1):
        if seqstr[i:i + motiflen] in motifs:
        
        	if singleton:
        		# checking the target motif not in the left and right stream
        		left_region = seqstr[max(i-10,0) : i]
        		right_region = seqstr[i+motiflen : min(i+motiflen+10, ref_seq_len)]

        		if ((motifs[0] not in left_region) and (motifs[0] not in right_region)):
        			sites.append(i + methy_shift)
        		else:
        			num_non_singleton += 1
        	else:
        		sites.append(i + methy_shift)

    #if singleton:
    #	print(" |-* [singleton mode] filtered non-singleton %d" %(num_non_singleton))

    return sites


""" input is the extracted re-squggiled sequence and signals"""
## add the position infomraiton here
def read_motif_input_extraction(signal, event, align_info, motif, m_shift, r_mer):

	#1. get candidate region context for the sequence of motif
	seq_event = "".join([e[2] for e in event])
	loc_event = [ (e[0],e[1]) for e in event ]

	feat_input = []
	motif_local_loci = get_methysite_in_ref(seq_event, motif, m_shift)
	
	shift = (r_mer - 1)//2
	for loci in motif_local_loci:

		try:
			start = loci - shift
			end = loci + shift + 1

			# skip the out of range ones
			if(start < 0 or end > len(seq_event)):
				continue
			
			## chrome_name, start, motif_relative_location, align_strand, strand_len, $$ new added read_id, strand_tempalte)
			## original used
			a_info = (align_info[0], align_info[1], loci, align_info[3], len(seq_event), align_info[4], align_info[2])
			
			# nucleotide sequence
			r_mer_seq =  seq_event[start:end]

			# signal sequence
			r_mer_signal = [ signal[ l[0]:l[0]+l[1] ] for l in loc_event[start:end]]

			# added the length normalization
			total_seg_len = sum([len(r) for r in r_mer_signal])
			event_stat   = [[np.mean(r), np.std(r), len(r)/total_seg_len] for r in r_mer_signal]

			#event_stat   = [[np.mean(r), np.std(r), len(r)] for r in r_mer_signal]

			c_signal  = _get_central_signals(r_mer_signal, rawsignal_num=360)

			# filtering out the extrem signals ...
			mean_s_min = min([s[0] for s in event_stat])
			mean_s_max = max([s[0] for s in event_stat])

			# filtering out the extrem values
			if mean_s_min < -10 or mean_s_max > 10:
				#print("min=%f, max=%f" %(mean_s_min, mean_s_max))
				continue

			# masking the according signals
			# event_stat[shift+m_shift]=[0,0,0]
		
			feat_input.append((r_mer_seq, c_signal, a_info, event_stat))

		except Exception:
			# when erorr exists, this information will not be enough for debug.
			print("[!] Skip one motif r-mer out of boundary.")
			
	return feat_input
		

# segment fast5 signals based on the statistics
# this code is based on the implementation os SquiggleKit
# not very useful in this application, as many hyper-parameters and noisy local signals
def do_segment(sig, window=150, std_scale=0.75, seg_dist=50, corrector_w=50, stall_len=0.25):

	mn, mx = sig.min(), sig.max()
	mean, stdev, mmedian = np.mean(sig), np.std(sig), np.median(sig)
	top, bot = median + (stdev * std_scale), median - (stdev * std_scale)

	prev = False
	err, prev_err,c = 0,0,0
	w = corrector_w
	start,end = 0, 0
	segs = []

	for i in range(len(sig)):
		a = sig[i]
		if a < top and a > bot: # in the range
			if not prev:
				start = i
				prev = True
			c += 1
			w += 1
			if prev_err:
				prev_err = 0
			if c >= window and c >= w and not c%w:
				err -= 1
		else:
			if prev and err < error:
				c += 1
				err += 1
				prev_err += 1
				if c >= window and c>=w and not c%w:
					err -= 1
			elif prev and (c >= window or not segs and c > window*stall_len):
				end = i - prev_err
				prev = False
				if segs and start -segs[-1][1] < seg_dist: # if segment very close merge, update the end of segment
					segs[-1][1] = end
				else:
					segs.append([start,end])
				c = 0
				err = 0
				prev_err = 0
			elif prev:
				prev = False
				c = 0
				err = 0
				prev_err = 0
			else:
				continue
	if segs:
		return segs
	else:
		return False

# visulaiation of the segment by SquiggleKit
def view_segs(segs, sig, savepath):
    fig = plt.figure(1)
    #fig.subplots_adjust(hspace=0.1, wspace=0.01)
    ax = fig.add_subplot(111)

    # Show segment lines
    for i, j in segs:
        ax.axvline(x=i, color='m')
        ax.axvline(x=j, color='m')

    plt.plot(sig, color='k')
    plt.savefig(savepath)
    plt.clf()



# single read input feature extraction
def single_read_process(file_name, motif, m_shift, w_len):

	try:
		# merge the following two functions 

		signal, events, align_info = get_signal_event_from_fast5(file_name)
		read_motif_input = read_motif_input_extraction(signal, events, align_info, motif, m_shift, w_len)

		return read_motif_input
	except:
		return None

	


