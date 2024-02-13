# Understanding the Role of the Projector in Knowledge Distillation
This is the official implementation of AAAI24 paper "Understanding the Role of the Projector in Knowledge Distillation"
Code for the AAAI24 paper:

```text
"Understanding the Role of the Projector in Knowledge Distillation".
Roy Miles, Krystian Mikolajczyk. AAAI 2024.
```
[[Paper on arxiv](https://arxiv.org/abs/2303.11098)]

## Structure

The two main sets of experiments corresponding to the DeIT and ResNet results can be found in folders `deit/` and `resnet/` respectively. The DeIT code is based on that provided by [Co-Advise](https://github.com/OliverRensu/co-advise), while the ResNet code uses the [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) library.


## Pretrained Models

We provide the pre-distilled model weights and logs for the DeIT experiments.

| model | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| tiny | 77.2 | 93.7 | 5M | [model](https://drive.google.com/drive/folders/1X8U2EA1gGtJ1RugbjUw6GM0-BGtEZ-LO?usp=drive_link) |
| small | 82.1 | 96.0 | 22M| [model](https://drive.google.com/drive/folders/1e6i7Aq_cKEMrGRuLP_9gt7LzcAQbeO0F?usp=drive_link) |

## Testing and Training

Before training, make sure to change the `deit/config.py` entries for your data path, output directory etc. The RegNet160 teacher weights are expected to be found in `ckpts/regnety_160-a5fe301d.pth`, else the most recent pre-trained Hugging Face weights will be downloaded. Note that the results reporting in our paper using the same teacher weights used by DeIT. We have tested training with 1 and 2 GPUs using effective batches sizes between 256 and 1024. Using larger batch sizes, or more GPUs, may require modifying the distributed training slightly and/or the learning rates.

Training is then simply run as follows:
```
python main.py --model tiny --train_student
```

Omitting the `--train_student` argument will evaluate the model using the checkpoint weights in `ckpts/ckpt_epoch_299.pth`. See `deit/main.py` for more details. 
```
python main.py --model tiny
```

## ImageNet

For training a ResNet18 student using a ResNet34 teacher, we use the `torchdistill` library.
```
cd imagenet
python image_classification.py --config configs/ilsvrc2012/ours/res3418.yaml --log log/output.txt
```

## Pretrained Models

We provide the pre-distilled model weights and logs. This reproduced experiment has an accuracy **higher** than that reported in the original paper.

| model | acc@1 | url |
| --- | --- | --- |
| resnet18 | 71.87 | [model](https://drive.google.com/drive/folders/1P5mePA0vwWkGqzJCiExfVzpqZEpEDEEz?usp=sharing) |

## Citation
```
@InProceedings{miles2023understanding_AAAI,
      title      = {Understanding the Role of the Projector in Knowledge Distillation}, 
      author     = {Roy Miles and Krystian Mikolajczyk},
      booktitle  = {Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI-24)},
      year       = {2023},
      month      = {December}
}
```

If you have any questions, feel free to email me! 