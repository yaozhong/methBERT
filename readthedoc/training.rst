.. _training:

Models and training
=====================================

Provided deep learning models
-------------------------------------
* biRNN
* bert
* bert_plus

We used two types of BERT model structures : the standard one [Devlin et al., 2018] and the refined one. Compared to the standard model structures,
the refined BERT considers task-specific features described as follows.

	- learnable postional embedding
	- self-attetion with realtive postion representation [Shaw et al.]
	- center postitions concatenation for the output layer.


Training models
-------------------------------------
Basic settings::

	N_EPOCH=50
	W_LEN=21
	LR=1e-4
	MOTIF="CG"
	NUCLEOTIDE_LOC_IN_MOTIF=0
	POSITIVE_SAMPLE_PATH=<methylated fast5 path>
	NEGATIVE_SAMPLE_PATH=<unmethylated fast5 path>
	MODEL_SAVE_PATH=<model saved path>

- BiRNN ::

	# training biRNN model
	MODEL="biRNN_basic"
	python3 train_biRNN.py --model ${MODEL}  --model_dir ${MODEL_SAVE_PATH} --gpu cuda:0 --epoch ${N_EPOCH} \
 	--positive_control_dataPath ${POSITIVE_SAMPLE_PATH}   --negative_control_dataPath ${NEGATIVE_SAMPLE_PATH} \
 	--motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --w_len ${W_LEN} --lr $LR \
 	--batch_size ${BATCH_SIZE}  --num_worker ${N_WORKER} --data_balance_adjust

- BERTs ::
	
	# training bert models and using randomly read selection
	MODEL="BERT_plus" (option: "BERT", "BERT_plus")
	python3 train_bert.py --model ${MODEL}  --model_dir ${MODEL_SAVE_PATH} --gpu cuda:0 --epoch ${N_EPOCH} \
 	--positive_control_dataPath ${POSITIVE_SAMPLE_PATH}   --negative_control_dataPath ${NEGATIVE_SAMPLE_PATH} \
 	--motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --w_len ${W_LEN} --lr $LR\
 	--batch_size ${BATCH_SIZE}  --num_worker ${N_WORKER} --data_balance_adjust


 	## training bert models and using region-based read selection
 	TEST_REGION="NC_000913.3 1000000 2000000"
 	python3 train_bert.py --model ${MODEL} --model_dir ${MODEL_SAVE_PATH} --gpu cuda:0 --epoch ${N_EPOCH} \
 	--positive_control_dataPath ${POSITIVE_SAMPLE_PATH}   --negative_control_dataPath ${NEGATIVE_SAMPLE_PATH} \
	--motif ${MOTIF} --m_shift  --w_len=${W_LEN} --lr $LR \
	--batch_size ${BATCH_SIZE} --num_worker ${N_WORKER} --data_balance_adjust \
	--test_region $TEST_REGION
