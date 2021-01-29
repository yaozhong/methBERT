# BERT model for nanopore methylation detection


![](figures/BERT_model_refined.eps)


## Docker enviroment
We provide a docker image for running this code
```
docker pull yaozhong/deep_intra_sv:0.9
```
* ubuntu 14.04.4
* Python 3.6




## Training

```
N_EPOCH=50
W_LEN=21
LR=1e-4
MODEL="biRNN_basic"
DATA="stoiber_ecoli"

DATA_EXTRA="M_Hhal_gCgc"
MODEL_PATH="/home/yaozhong/working/2_nanopore/methylation/experiment/meth_baseline/${MODEL}_W${W_LEN}_E${N_EPOCH}_${DATA}-${DATA_EXTRA}_basic_lr-${LR}_ttt.pth"
LOG="/home/yaozhong/working/2_nanopore/methylation/experiment/run_log/20210104/${MODEL}_W${W_LEN}_E${N_EPOCH}_${DATA}-${DATA_EXTRA}_10batch_lr-${LR}.txt"

echo "= Training $MODEL with stacked substract 7-feat on $DATA, $DATA_EXTRA="
python3 train.py --model $MODEL  --model_dir ${MODEL_PATH} --gpu cuda:0 --epoch ${N_EPOCH} \
 --dataset $DATA --dataset_extra $DATA_EXTRA --motif GCGC --m_shift 1 --w_len ${W_LEN} --lr $LR  | tee $LOG

```

## Detection
```
# Example
N_EPOCH=50
W_LEN=21
LR=1e-4

DATA="stoiber_ecoli"
DATA_EXTRA="M_Sssl_Cg"

MODEL="BERT_plus" 
MODEL_PATH="/home/yaozhong/working/2_nanopore/methylation/experiment/paper_model/${MODEL}_W${W_LEN}_E${N_EPOCH}_${DATA}-${DATA_EXTRA}_linear-7x_tanh.pth"

split="m80"

REF="/home/yaozhong/working/2_nanopore/methylation/data/ref/ecoli_k12.fasta"
OUTPUT="/home/yaozhong/working/2_nanopore/methylation/experiment/paper_results/dataset1_${split}_${MODEL}-${DATA}-${DATA_EXTRA}_time.tsv"
FAST5_FOLD="/home/yaozhong/working/2_nanopore/methylation/data/Simpson/benchmark/dataset1/${split}"

time python detect.py --model $MODEL --model_dir $MODEL_PATH \
--gpu cuda:0 --dataset $DATA --dataset_extra $DATA_EXTRA  \
--fast5_fold $FAST5_FOLD --num_worker 24 \
--motif CG --m_shift 0 --evalMode test_mode --w_len 21 --ref_genome $REF --output_file $OUTPUT
```
