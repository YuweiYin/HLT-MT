# HLT-MT: High-resource Language-specific Training for Multilingual Neural Machine Translation

![picture](https://yuweiyin.github.io/files/publications/2022-07-23-IJCAI-MNMT-HLT.png)

## Abstract

Multilingual neural machine translation (MNMT)
trained in multiple language pairs has attracted considerable
attention due to fewer model parameters
and lower training costs by sharing knowledge
among multiple languages. Nonetheless, multilingual
training is plagued by language interference
degeneration in shared parameters because
of the negative interference among different translation
directions, especially on high-resource languages.
In this paper, we propose the multilingual
translation model with the high-resource language-specific
training (**HLT-MT**) to alleviate the negative
interference, which adopts the two-stage training
with the language-specific selection mechanism.
Specifically, we first train the multilingual
model only with the high-resource pairs and select
the language-specific modules at the top of
the decoder to enhance the translation quality of
high-resource directions. Next, the model is further
trained on all available corpora to transfer knowledge
from high-resource languages (HRLs) to low-resource
languages (LRLs). Experimental results
show that HLT-MT outperforms various strong
baselines on WMT-10 and OPUS-100 benchmarks.
Furthermore, the analytic experiments validate the
effectiveness of our method in mitigating the negative
interference in multilingual training.

## Data

* **WMT-10**
  * Multilingual dataset (from the [WMT corpus](https://www.statmt.org/)) with 11 languages: 10 English-centric language pairs.
  * English (En), French (Fr), Czech (Cs), German(De), Finnish (Fi), Latvian (Lv), Estonian (Et), Romanian (Ro), Hindi(Hi), Turkish (Tr), and Gujarati (Gu).
* **OPUS-100**
  * Massively multilingual dataset (from the [OPUS-100 corpus](https://opus.nlpl.eu/opus-100.php)) with 100 languages.
  * 94 English-centric language pairs are used after dropping out 5 languages that lack corresponding test sets.

## Environment

* Python: >= 3.6
* [PyTorch](http://pytorch.org/): >= 1.5.0
* NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* [Fairseq](https://github.com/pytorch/fairseq): 1.0.0

```bash
cd HLT-MT/fairseq
pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```


## HLT-MT Training

```bash
# parameters
NODES=${1}
MAX_TOKENS=${2}
UPDATE_FREQ=${3}
MAX_EPOCH=${4}
LR=${5}
WARMUP_STEPS=${6}
WEIGHT_DECAY=${7}
HIGH_LANGS=${8}
LOW_LANGS=${9}
LANG_PAIRS=${10}
ADAPTER_NUM=${11}
ADAPTER_DIM=${12}
APPEND_CMDS=${13}

# default values
if [ ! ${NODES} ]; then
    NODES=4
fi
if [ ! ${MAX_TOKENS} ]; then
    MAX_TOKENS=4096
fi
if [ ! ${UPDATE_FREQ} ]; then
    UPDATE_FREQ=4
fi
if [ ! ${LR} ]; then
    LR=3e-4
fi
if [ ! ${WARMUP_STEPS} ]; then
    WARMUP_STEPS=4000
fi
if [ ! ${WEIGHT_DECAY} ]; then
    WEIGHT_DECAY=0
fi
if [ ! ${ADAPTER_NUM} ]; then
    ADAPTER_NUM=3
fi
if [ ! ${ADAPTER_DIM} ]; then
    ADAPTER_DIM=3072
fi

LANGS="en,fr,cs,de,fi,lv,et,ro,hi,tr,gu"
GPUS=8
bsz=$((${MAX_TOKENS}*${UPDATE_FREQ}*${NODES}*${GPUS}))

TEXT=/path/to/data-bin/
MODEL=/path/to/model/
PRETRAINED_ENCODER_MODEL=/path/to/xlmr_model

python -m torch.distributed.launch \
  --nproc_per_node=${GPUS} --nnodes=${NODES} --node_rank=${OMPI_COMM_WORLD_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train.py ${TEXT} \
  --save-dir ${MODEL} --arch "two_stage_sparse_transformer" \
  --variant addffn --pretrained-infoxlm-checkpoint ${PRETRAINED_ENCODER_MODEL} \
  --init-encoder-only --init-decoder-only --task "translation_multi_simple_epoch" \
  --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 \
  --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' --langs ${LANGS} \
  --high-langs ${HIGH_LANGS} --low-langs ${LOW_LANGS} --lang-pairs ${LANG_PAIRS} \
  --ddp-backend=no_c10d --enable-reservsed-directions-shared-datasets \
  --share-all-embeddings --max-source-positions 256 --max-target-positions 256 \
  --criterion "label_smoothed_cross_entropy_with_sparse" --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr ${LR} \
  --warmup-epoch 5 --warmup-updates 4000 \
  --max-update 400000 --max-epoch ${MAX_EPOCH} --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
  --dropout 0.1 --attention-dropout 0.0 --weight-decay ${WEIGHT_DECAY} \
  --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test \
  --fp16 --truncate-source --same-lang-per-batch --enable-lang-ids \
  --use-adapter --adapter-num ${ADAPTER_NUM} --adapter-dim ${ADAPTER_DIM} \
  --swap-adapter 0 --start-hard-epoch 1 --hard-adapter 0.5 --end-soft-epoch 5 --disparity-weight 1.0 \
  --log-file ${MODEL}/train.log --tensorboard-logdir ${MODEL}/logs ${APPEND_CMDS}
```


## Evaluation

* **Metrics**: the case-sensitive detokenized BLEU using sacreBLEU:
  * BLEU+case.mixed+lang.{src}-{tgt}+numrefs.1+smooth.exp+tok.13a+version.1.3.1


## Citation

<!-- Paper Link:  -->

```bibtex
@inproceedings{hlt-mt,
  title = {High-resource Language-specific Training for Multilingual Neural Machine Translation},
  author = {Jian Yang and Yuwei Yin and Shuming Ma and Dongdong Zhang and Zhoujun Li and Furu Wei}
  booktitle = {IJCAI 2022},
  year = {2022},
}
```


## License

Please refer to the [LICENSE](./LICENSE) file for more details.


## Contact

If there is any question, feel free to create a GitHub issue or contact us by [Email](mailto:seckexyin@gmail.com).
