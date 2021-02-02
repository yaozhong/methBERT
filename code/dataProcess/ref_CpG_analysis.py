#!/usr/bin/python

""" Do -CpG- analysis for the reference genome """
import collections, re
import matplotlib.pyplot as plt

def ref_fasta_load(ref_file):

	ref, name = [], []
	genome = ''
	with open(ref_file, 'r') as f:
		for line in f:
			if line[0] == '>':
				name.append(line[1:-1].split(" ")[0])
				if len(genome) > 0 :
					ref.append(genome)
				genome = ''
			else:
				genome += line.rstrip()

		ref.append(genome)

	assert(len(ref) == len(name))
	print(name)

	return zip(name, ref)


def cpg_find(ref_file):

	ref = ref_fasta_load(ref_file)

	kmers = []
	kmer_position = []
	for n, seq in ref:
		motif = r'\w{2}CG\w{2}'
		kmers.extend(re.findall(motif, seq))
		kmer_position.extend([ (n, m.start(), m.end()) for m in re.compile(motif).finditer(seq)])

	w = (collections.Counter(kmers))
	#plt.bar(w.keys(), w.values())
	#plt.show()


if __name__ == "__main__":

	ecoli_ref = "../../data/ref/ecoli_k12.fasta"
	human_ref = "../../data/ref/GRCh38.primary_assembly.genome.fa"

	cpg_find(human_ref)