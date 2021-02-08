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
POSITIVE_SAMPLE_PATH=<methylated fast5 path>
NEGATIVE_SAMPLE_PATH=<unmethylated fast5 path>
MODEL_SAVE_PATH=<model saved path>

# training biRNN model
python3 train_biRNN.py --model ${MODEL}  --model_dir ${MODEL_SAVE_PATH} --gpu cuda:0 --epoch ${N_EPOCH} \
 --positive_control_dataPath ${POSITIVE_SAMPLE_PATH}   --negative_control_dataPath ${NEGATIVE_SAMPLE_PATH} \
 --motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --w_len ${W_LEN} --lr $LR 

# training bert models
MODEL="BERT_plus" (option: "BERT", "BERT_plus")
python3 train_bert.py --model ${MODEL}  --model_dir MODEL_SAVE_PATH --gpu cuda:0 --epoch ${N_EPOCH} \
 --positive_control_dataPath ${POSITIVE_SAMPLE_PATH}   --negative_control_dataPath ${NEGATIVE_SAMPLE_PATH} \
 --motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --w_len ${W_LEN} --lr $LR 
```

## Detection

We provided independent trained models on each 5mC and 6mA datasets of different motifs and methyltransferases in the ./trained_model fold.

```
MODEL="BERT_plus" 
MODEL_SAVE_PATH=<model saved path>
REF=<reference genome fasta file>
FAST5_FOLD=<fast5 files to be analyzed>
OUTPUT=<output file>

time python detect.py --model ${MODEL} --model_dir ${MODEL_SAVE_PATH} \
--gpu cuda:0  --fast5_fold ${FAST5_FOLD} --num_worker 12 \
--motif ${MOTIF} --m_shift ${NUCLEOTIDE_LOC_IN_MOTIF} --evalMode test_mode --w_len ${W_LEN} --ref_genome ${REF} --output_file ${OUTPUT}
```

We generate the same output format as the deepSignal (https://github.com/bioinfomaticsCSU/deepsignal).

```
# output example
NC_000913.3     4581829 +       4581829 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     3.0398369e-06   0       TGCGGGTCTTCGCCATACACG
NC_000913.3     4581838 +       4581838 43ea7b03-8d2b-4df3-b395-536b41872137    t       0.9999996       0.00013372302   0       TCGCCATACACGCGCTCAAAC
NC_000913.3     4581840 +       4581840 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     0.0     0       GCCATACACGCGCTCAAACGG
NC_000913.3     4581848 +       4581848 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     0.0     0       CGCGCTCAAACGGCTGCAAAT
NC_000913.3     4581862 +       4581862 43ea7b03-8d2b-4df3-b395-536b41872137    t       1.0     0.0     0       TGCAAATGCTCGTCGGTAAAC
```


## Available benchmark dataset

We test models on 5mC and 6mA dataset sequenced with Nanopore R9 flow cells, 
which is commonly used as the benchmark data in the previous work.

- The stoiber's dataset, https://www.biorxiv.org/content/10.1101/094672v2
- Simpson's dataset, https://www.nature.com/articles/nmeth.4184

The fast5 reads are supposed to be pre-processed with re-squggle ([Tombo](https://github.com/nanoporetech/tombo)) 
```
tombo resquiggle --dna $FAST5_FOLD $REF --processes 24 --corrected-group RawGenomeCorrected_001 --basecall-group Basecall_1D_000 
```

### Reference genome
- E.coli: K-12 sub-strand MG1655
- H.sapiens: GRCh38


## Reference
This source code is refering to the follow github projects. 
- [DeepMOD](https://github.com/WGLab/DeepMod)
- [DeepSignal](https://github.com/bioinfomaticsCSU/deepsignal)
- [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)



