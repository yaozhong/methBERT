.. _Dataset

Benchmark dataset
============================

Currently, we only tested the bert models on the benchmark R9 dataset provided by Stoiber (2016) and Simpson (2017).
The proposed framework can be applied to other flowcells when positive and negative fast5 files are provided.

Stoiber's R9 dataset
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

Simpson's R9 dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* 5mC methylation (https://www.nature.com/articles/nmeth.4184)
	*  PRJEB13021<http://www.ebi.ac.uk/ena/data/view/PRJEB13021>`__
* Control (E.coli K12 PCR-amplicon)
	* `ERR1147229 <http://www.ebi.ac.uk/ena/data/view/ERR1147229>`__

