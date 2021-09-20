.. _data_processing:

Data
=====================================

Data preprocessing
-------------------------------------
Fast5 files of reads are first preprocessed to generate inputs for neural networks.
Before that, we assumed each read is already basecalled and aligned to a reference genome.
The basecalls, events and alignments are saved in the "Analyses" group in the fast5 file.

If starting from raw fast5 reads, the following tools can be used.::

	# basecalling (Guppy >=3.2)
	guppy_basecaller -c <CONFIG.file> -i <Fast5 fold> -s <Output fold> -x cuda:all

	# alignement (minimap2)
	REF=<Reference genome>
	minimap2 -a -x map-ont $REF output.fastq | samtools sort -T tmp -o output.sorted.bam 
	samtools index output.sorted.bam 

	# re-squggle (Tombo)
	tombo resquiggle <fast5_fold> <Ref_genome_fasta> --processes <num_worker> --corrected-group RawGenomeCorrected_001 --basecall-group Basecall_1D_000 --overwrite

Fast5 files can be mannually investigated using HDFView(https://www.hdfgroup.org/downloads/hdfview/).


Data sampling and split
-------------------------------------
After raw fast5 reads are pre-processed, each read will be used to generate training and testing data
for target motifs (e.g., CpG).
Reads of complete methylation data and amplicon data (Control) are used as positive and negative samples, respectively.
To accelerate the data generating process, we first utilize multi-cores loading processed data into RAM and cached.
In this stage, a read is a basic unit for preprocessing.
In the training stage, a screening window surrounding the target motif is the basic unit for the batch training. 
If any GPU is available, batched screening windows are transferred from RAM to GPU memory during training, assuming GPU memory is limited.

We provide the following two options for generating training and testing data.

1. *random and balanced selection according to the number of reads*::

	--data_balance_adjust

2. *region-based selection*

  Provide the region used for the testing with ---test_region option::

  	--test_region NC_000913.3 1000000 2000000  


Used R9 Benchmark dataset
-------------------------------

Currently, we only tested the bert models on the benchmark R9 dataset provided by Stoiber (2016) and Simpson (2017)


Stoiber's R9 training dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Control (PCR-amplicon) 

	* `Con1 <http://s3.climb.ac.uk/nanopore-methylation/Control_lib1.tar>`__
	* `Con2 <http://s3.climb.ac.uk/nanopore-methylation/Control_lib3.tar>`__ 

* 6mA methylation

	* `TCG[A], Taql <http://s3.climb.ac.uk/nanopore-methylation/meth1_lib1.tar>`__
	* `GA[A]TTC, EcoRI <http://s3.climb.ac.uk/nanopore-methylation/meth4_lib1.tar>`__
	* `G[C]GC, Dam <http://s3.climb.ac.uk/nanopore-methylation/meth8_lib2.tar>`__

* 5mC methylaiton

	* `[C]G, MpeI <http://s3.climb.ac.uk/nanopore-methylation/meth9_lib2.tar>`__
	* `[C]G, SssI <http://s3.climb.ac.uk/nanopore-methylation/meth10_lib3.tar>`__
	* `G[A]TC, HhaI <http://s3.climb.ac.uk/nanopore-methylation/meth11_lib3.tar>`__

Simpson's R9 training dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~