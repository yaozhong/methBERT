""" 
Data pre-processing functions in this file are used from deepSignal modules.
(https://github.com/bioinfomaticsCSU/deepsignal)
"""



import sys, os, math
import fnmatch
import numpy as np
import random
import h5py
import scrappy
import csv
import torch


def _get_readid_from_fast5(h5file, reads_group = 'Raw/Reads'):
    first_read = list(h5file[reads_group].keys())[0]
    if sys.version_info[0] >= 3:
        try:
            read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'], 'utf-8')
        except TypeError:
            read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'])
    else:
        read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'])

    return read_id


def _get_alignment_attrs_of_each_strand(strand_path, h5obj):
    strand_basecall_group_alignment = h5obj['/'.join([strand_path, 'Alignment'])]
    alignment_attrs = strand_basecall_group_alignment.attrs
    # attr_names = list(alignment_attrs.keys())

    if strand_path.endswith('template'):
        strand = 't'
    else:
        strand = 'c'
    if sys.version_info[0] >= 3:
        try:
            alignstrand = str(alignment_attrs['mapped_strand'], 'utf-8')
            chrom = str(alignment_attrs['mapped_chrom'], 'utf-8')
        except TypeError:
            alignstrand = str(alignment_attrs['mapped_strand'])
            chrom = str(alignment_attrs['mapped_chrom'])
            if chrom.startswith('b'):
                chrom = chrom.split("'")[1]
    else:
        alignstrand = str(alignment_attrs['mapped_strand'])
        chrom = str(alignment_attrs['mapped_chrom'])
        if chrom.startswith('b'):
            chrom = chrom.split("'")[1]
    chrom_start = alignment_attrs['mapped_start']

    # whether the alignment score can be acuired?

    return strand, alignstrand, chrom, chrom_start


def _get_alignment_info_from_fast5(fast5_path, corrected_group='RawGenomeCorrected_001',
                                   basecall_subgroup='BaseCalled_template'):
    try:
        h5file = h5py.File(fast5_path, mode='r')
        corrgroup_path = '/'.join(['Analyses', corrected_group])

        if '/'.join([corrgroup_path, basecall_subgroup, 'Alignment']) in h5file:
            readname = _get_readid_from_fast5(h5file)
            strand, alignstrand, chrom, chrom_start = _get_alignment_attrs_of_each_strand('/'.join([corrgroup_path,
                                                                                                    basecall_subgroup]),
                                                                                          h5file)
            h5file.close()
            return readname, strand, alignstrand, chrom, chrom_start
        else:
            return '', '', '', '', ''
    except IOError:
        print("the {} can't be opened".format(fast5_path))
        return '', '', '', '', ''


def get_fast5s(fast5_dir, is_recursive=True):
    fast5_dir = os.path.abspath(fast5_dir)
    fast5s = []
    if is_recursive:
        for root, dirnames, filenames in os.walk(fast5_dir):
            for filename in fnmatch.filter(filenames, '*.fast5'):
                fast5_path = os.path.join(root, filename)
                fast5s.append(fast5_path)
    else:
        for fast5_name in os.listdir(fast5_dir):
            if fast5_name.endswith('.fast5'):
                fast5_path = '/'.join([fast5_dir, fast5_name])
                fast5s.append(fast5_path)
    return fast5s


def _get_central_signals(signals_list, rawsignal_num=360):
    signal_lens = [len(x) for x in signals_list]

    if sum(signal_lens) < rawsignal_num:
        # real_signals = sum(signals_list, [])
        real_signals = np.concatenate(signals_list)
        cent_signals = np.append(real_signals, np.array([0] * (rawsignal_num - len(real_signals))))
    else:
        mid_loc = int((len(signals_list) - 1) / 2)
        mid_base_len = len(signals_list[mid_loc])

        if mid_base_len >= rawsignal_num:
            allcentsignals = signals_list[mid_loc]
            cent_signals = [allcentsignals[x] for x in sorted(random.sample(range(len(allcentsignals)),
                                                                            rawsignal_num))]
        else:
            left_len = (rawsignal_num - mid_base_len) // 2
            right_len = rawsignal_num - left_len

            left_signals = np.concatenate(signals_list[:mid_loc])
            right_signals = np.concatenate(signals_list[mid_loc:])

            if left_len > len(left_signals):
                right_len = right_len + left_len - len(left_signals)
                left_len = len(left_signals)
            elif right_len > len(right_signals):
                left_len = left_len + right_len - len(right_signals)
                right_len = len(right_signals)

            assert (right_len + left_len == rawsignal_num)
            if left_len == 0:
                cent_signals = right_signals[:right_len]
            else:
                cent_signals = np.append(left_signals[-left_len:], right_signals[:right_len])

    return cent_signals

# generate signal for the given sequence, scrappie, taiyaki has a newer version
# please checking the details of parameters before the final usage
def sim_seq_singal(seq, mod="squiggle_r94"):

    gen_signal = []
    signal = scrappy.sequence_to_squiggle(seq, model=mod).data(as_numpy=True, sloika=False)
    for i in signal:
        gen_signal.append([i[0], np.exp(i[1]), np.exp(-i[2])])
    return gen_signal

# use the 6-mer model to generate the position seperately
def load_poreModel(modelPath="../data/kmer_models/r9.4_180mv_450bps_6mer/template_median68pA.model"):

    poreModel = {}

    with open(modelPath, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader)
        for line in csv_reader:
         poreModel[line[0]] = line[1:]

    print(len(poreModel.keys()))

    return poreModel


def sim_seq_singal_with_PoreModel(seq, mod="squiggle_r94"):

    seq_len = len(seq)
    gen_signal = []

    for i in range(seq_len-6):
        s = seq[i:i+6]
        signal = scrappy.sequence_to_squiggle(s, model=mod).data(as_numpy=True, sloika=False)
        gen_signal.append([np.mean(signal[:,0]), np.exp(np.mean(signal[:,1])), np.exp(-np.sum(signal[:,2]))])

    for i in range(6):
        gen_signal.append([0,0,0])

    return gen_signal


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


class DNAReference:
    def __init__(self, reffile):
        self._contignames = []
        self._contigs = {}  # contigname 2 contigseq
        with open(reffile, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        self._contigs[contigname] = contigseq
                        self._contignames.append(contigname)
                    contigname = line.strip()[1:].split(' ')[0]
                    contigseq = ''
                else:
                    # turn to upper case
                    contigseq += line.strip().upper()
            self._contigs[contigname] = contigseq
            self._contignames.append(contigname)

    def getcontigs(self):
        return self._contigs

    def getcontignames(self):
        return self._contignames


def get_contig2len(ref_path):
    refseq = DNAReference(ref_path)
    chrom2len = {}
    for contigname in refseq.getcontignames():
        chrom2len[contigname] = len(refseq.getcontigs()[contigname])
    del refseq
    return chrom2len


if __name__ == "__main__":
    seq = "AAGGCTAGCTAGCT"
    #signal = sim_seq_singal(seq)
    #print(signal)

    # testing loading k-mer model
    modelPath="/Users/yaozhong/Research/202003_nanopore_Methylation/data/kmer_models/r9.4_180mv_450bps_6mer/template_median68pA.model"
    load_poreModel(modelPath)

    # get the reference genome length
    chrom2len = get_contig2len(reference_path)
    

