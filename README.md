# MEPDNet-Pytorch

PyTorch implementation of multi-encoder parse-decoder network for sequential medical image segmentation.

## Train

```bash
python run.py --model $MODEL_NAME --mode train -l $LR -b $BATCH_SIZE -e $EPOCHS --gpu-id $GPU_ID

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --mode {train,test,use}
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -l LR, --learning-rate LR
                        Learning rate
```
See examples in [train.sh](train.sh).


## Test

```bash
python run.py --model $MODEL_NAME --mode test --state $MODEL_ID -b $BATCH_SIZE --gpu-ids $GPU_ID

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --mode {train,test,use}
  --gpu-ids GPU_IDS [GPU_IDS ...]
  --state STATE
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
```
See examples in [test.sh](test.sh).

