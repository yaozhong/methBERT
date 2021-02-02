""" 20200426 event k-mer assembly """

def adjacent_overlap(k1, k2):

	idx = -1
	# start position
	for i in range(len(k1)):
		if k1[i:] == k2[:len(k2)-i]:
			idx = i
			break

	return idx

def kmer_simple_assembly(k_mer):

	non_overlap = [k_mer[0]]

	for i in range(len(k_mer)-1):
		idx = adjacent_overlap(k_mer[i], k_mer[i+1])
		if idx > 0 :
			non_overlap.append(k_mer[i+1][-idx:])

	contig = "".join(non_overlap)

	return contig


if __name__ == "__main__":

	k_mer = ['AAAAC', 'AAACA', 'AAACA', 'AAACA', 'AAACA', 'AACAC', 'ACACT', 'ACACT', 'ACACT']
	contig = kmer_simple_assembly(k_mer)

	print(contig)