# BERT model for nanopore methylation detection


![](figures/BERT_model_refined.png)

## Docker enviroment
We provide a docker image for running this code
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
MODEL="biRNN_basic" ("BERT", "BERT_plus")
MOTIF="CG"
NUCLEOTIDE_LOC_IN_MOTIF=0

python3 train.py --model ${MODEL}  --model_dir MODEL_SAVE_PATH --gpu cuda:0 --epoch ${N_EPOCH} \
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
