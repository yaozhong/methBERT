.. _testing:

Evaluation and methylation prediction
=======================

Evaluation types
-----------------------
* single-read level
* genomic-loci (group-reads) level


Evaluation metrics
-----------------------
* ROC-AUC
* PR-AUC


Methylation prediction
-----------------------

We provided independent trained models on each 5mC and 6mA datasets of different motifs and methyltransferases in the ./trained_model fold.::


	MODEL="BERT_plus" 
	MODEL_SAVE_PATH=<model saved path>
	REF=<reference genome fasta file>
	FAST5_FOLD=<fast5 files to be analyzed>
	OUTPUT=<output file>

	time python detect.py --model ${MODEL} --model_dir ${MODEL_SAVE_PATH} \
	--gpu cuda:0  --fast5_fold ${FAST5_FOLD} --num_worker 12 \
	--motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --evalMode test_mode --w_len ${W_LEN} --ref_genome ${REF} --output_file ${OUTPUT}


We generate the same output format as the deepSignal (https://github.com/bioinfomaticsCSU/deepsignal).::


	# output example
	NC_000913.3     4581829 +       4581829 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     3.0398369e-06   0       TGCGGGTCTTCGCCATACACG
	NC_000913.3     4581838 +       4581838 43ea7b03-8d2b-4df3-b395-536b41872137    t       0.9999996       0.00013372302   0       TCGCCATACACGCGCTCAAAC
	NC_000913.3     4581840 +       4581840 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     0.0     0       GCCATACACGCGCTCAAACGG
	NC_000913.3     4581848 +       4581848 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     0.0     0       CGCGCTCAAACGGCTGCAAAT
	NC_000913.3     4581862 +       4581862 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     0.0     0       TGCAAATGCTCGTCGGTAAAC

