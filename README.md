# Understanding the Role of the Projector in Knowledge Distillation
This is the official implementation of AAAI23 paper "Understanding the Role of the Projector in Knowledge Distillation"

## Structure

The two main sets of experiments corresponding to the DeIT and ResNet results can be found in folders `deit/` and `resnet/` respectively.
The DeIT code is based on that provided by [Co-Advise](https://github.com/OliverRensu/co-advise), while the ResNet code uses the [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) library.


## Pretrained Models

We provide the pre-distilled model weights and logs for the DeIT experiments.

| model | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| tiny | 77.2 | 93.7 | 5M | [model](https://drive.google.com/drive/folders/1X8U2EA1gGtJ1RugbjUw6GM0-BGtEZ-LO?usp=drive_link) |
| small | 82.1 | 96.0 | 22M| [model](https://drive.google.com/drive/folders/1e6i7Aq_cKEMrGRuLP_9gt7LzcAQbeO0F?usp=drive_link) |

## Citation
```
@InProceedings{miles2023understanding_AAAI,
      title      = {A closer look at the training dynamics of knowledge distillation}, 
      author     = {Roy Miles and Krystian Mikolajczyk},
      booktitle  = {Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI-24)},
      year       = {2023},
      month      = {December}
}
```

If you have any questions, feel free to email me! 