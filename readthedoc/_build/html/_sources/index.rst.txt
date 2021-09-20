.. methBERT documentation master file, created by
   sphinx-quickstart on Thu Jun 10 15:20:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to methBERT's documentation
====================================

MethBERT explores a non-recurrent modeling approach for nanopore methylation detection based on the bidirectional encoder representations from transformers (BERT). 
Compared with the state-of-the-art model using bi-directional recurrent neural networks (RNN), BERT can provide a faster model inference solution without the limit of computation in sequential order. 
We proviede two types of BERTs: the basic one [Devlin et al.] and the refined one. 
The refined BERT is refined according to the task-specific features, including:

* learnable postional embedding
* self-attetion with realtive postion representation [Shaw et al.]
* center postitions concatenation for the output layer

.. image:: figures/BERT_model_refined.png


(Currently, we only trained on the R9 benchmark data. R9.4.1 and R10.3 models will be provided in the next update,
when our data is ready.)


.. toctree::
   :hidden:
   
   installation
   data_processing
   training
   testing

Reference
-------------------------------------
* Yao-zhong Zhang et al., `On the application of BERT models for nanopore methylation detection <https://www.biorxiv.org/content/10.1101/2021.02.08.430070v1>`_
* Liu et al. `DeepMod <https://github.com/WGLab/DeepMod>`_
* Devlin et al., `<https://arxiv.org/pdf/1810.04805.pdf>`_
* Shaw et al., `<https://arxiv.org/pdf/1803.02155.pdf>`_



