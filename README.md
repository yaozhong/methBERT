# On the application of BERT models for nanopore methylation detection

![](figures/BERT_model_refined.png)

We explore a non-recurrent modeling approach for nanopore methylation detection based on the bidirectional encoder representations from transformers (BERT).
Compared with the state-of-the-art model with bi-directional recurrent neural networks (RNN), BERT can provide a faster model inference solution without the limit of
the sequential computation order.
We use two types of BERTs: the basic one [Devlin et al.] and refined one.
The refined BERT is refined according to the task-specific features described as follows.

- learnable postional embedding
- self-attetion with realtive postion representation [Shaw et al.]
- center postitions concatenation for the output layer

The model structures are shown in the above figure. 

## Docker enviroment
We provide a docker image for running this source code
```
docker pull yaozhong/ont_methylation:0.6
```
* ubuntu 14.04.4
* Python 3.5.2
* Pytorch 1.5.1+cu101
```
nvidia-docker run -it --shm-size=64G -v LOCAL_DATA_PATH:MOUNT_DATA_PATH yaozhong/ont_methylation:0.6
```

## Training
```
N_EPOCH=50
W_LEN=21
LR=1e-4
MODEL="biRNN_basic"
MOTIF="CG"
NUCLEOTIDE_LOC_IN_MOTIF=0

# training biRNN model
python3 train_biRNN.py --model ${MODEL}  --model_dir MODEL_SAVE_PATH --gpu cuda:0 --epoch ${N_EPOCH} \
 --positive_control_dataPath POSITIVE_SAMPLE_PATH   --negative_control_dataPath NEGATIVE_SAMPLE_PATH \
 --motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --w_len ${W_LEN} --lr $LR 

 # training bert models
MODEL="BERT_plus" (option: "BERT", "BERT_plus")
python3 train_bert.py --model ${MODEL}  --model_dir MODEL_SAVE_PATH --gpu cuda:0 --epoch ${N_EPOCH} \
 --positive_control_dataPath POSITIVE_SAMPLE_PATH   --negative_control_dataPath NEGATIVE_SAMPLE_PATH \
 --motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --w_len ${W_LEN} --lr $LR 
```

## Detection
```
N_EPOCH=50
W_LEN=21
LR=1e-4
MODEL="BERT_plus" 
REF="data/ref/ecoli_k12.fasta"
MOTIF="CG"
NUCLEOTIDE_LOC_IN_MOTIF=0

time python detect.py --model ${MODEL} --model_dir ${MODEL_PATH} \
--gpu cuda:0  --fast5_fold FAST5_FOLD --num_worker 12 \
--motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --evalMode test_mode --w_len ${W_LEN} --ref_genome ${REF} --output_file OUTPUT
```

## Avaialbe benchmark dataset

We test models on 5mC and 6mA dataset sequenced with Nanopore R9 flow cells, 
which is commonly used as the benchmark data in the previous work.

- The stoiber's dataset, https://www.biorxiv.org/content/10.1101/094672v2
- Simpson's dataset, https://www.nature.com/articles/nmeth.4184

The fast5 reads are supposed to be pre-processed with re-squggle ([Tombo](https://github.com/nanoporetech/tombo)) 


## Reference
This source code is refering to the follow github projects. 
- [DeepMOD](https://github.com/WGLab/DeepMod)
- [DeepSignal](https://github.com/bioinfomaticsCSU/deepsignal)
- [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)



