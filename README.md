# AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation

This codebase is the official implementation of [`AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation`](https://arxiv.org/abs/2304.12566) (ICML, 2023). 
This codebase is mainly based on [DomainBed](https://github.com/facebookresearch/DomainBed), and [T3A](https://github.com/matsuolab/T3A).

## Installation


## Quick start
#### (1) Downlload the datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path --dataset pacs
```
Note: change `--dataset pacs` for downloading other datasets (e.g., `vlcs`, `office_home`, `terra_incognita`). 


#### (2) Train a model on source domains
```sh
python -m domainbed.scripts.train\
       --data_dir /my/datasets/path\
       --output_dir /my/pretrain/path\
       --algorithm ERM\
       --dataset PACS\
       --hparams "{\"backbone\": \"resnet50\"}" 
```
This scripts will produce new directory `/my/pretrain/path`, which include the full training log. 

Note: change `--dataset PACS` for training on other datasets (e.g., `VLCS`, `OfficeHome`, `TerraIncognita`). 

Note: change `--hparams "{\"backbone\": \"resnet50\"}"` for using other backbones (e.g., `resnet18`, `ViT-B16`, `HViT`). 


#### (3) Evaluate model with test time adaptation
```sh
python -m domainbed.scripts.unsupervised_adaptation\
       --input_dir=/my/pretrain/path\
       --adapt_algorithm=T3A
```
This scripts will produce a new file in `/my/pretrain/path`, whose name is `results_{adapt_algorithm}.jsonl`. 

Note: change `--adapt_algorithm=T3A` for using other test time adaptation methods (`AdaNPC`, `AdaNPCBN`, or `TentClf`). 



#### (4) Evaluate model with fine-tuning classifier
```sh
python -m domainbed.scripts.supervised_adaptation\
       --input_dir=/my/pretrain/path\
       --ft_mode=clf
```
This scripts will produce a new file in `/my/pretrain/path`, whose name is `results_{ft_mode}.jsonl`. 


## Available backbones

* resnet18
* resnet50
* BiT-M-R50x3
* BiT-M-R101x3
* BiT-M-R152x2
* ViT-B16
* ViT-L16
* DeiT
* Hybrid ViT (HViT)
* MLP-Mixer (Mixer-L16)

## Reproducing results
#### Table 1 and Figure 2 (Tuned ERM and CORAL)

You can use `scripts/hparam_search.sh`. Specifically, for each dataset and base algorithm, you can just type a following command.
```
sh scripts/hparam_search.sh resnet50 PACS ERM
```
Note that, it automatically starts 240 jobs, and take many times to finish. 


#### Table 2 and Figure 1 (ERM with various backbone)

You can use `scripts/launch.sh`. Specifically, for each backbone, you can just type following commands. 

Specifically, for baselines based on ResNet-50 (PLClf, PLFull, SHOT, SHOTIM, T3A)

```
sh scripts/launch.sh pretrain resnet50 10 3 local ERM
sh scripts/launch.sh sup resnet50 10 3 local ERM
sh scripts/launch.sh unsup resnet50 10 3 local ERM
```

for baselines based on ResNet-50-BN (TentClf, TentNorm, TentFull)

```
sh scripts/launch.sh pretrain resnet50-BN 10 3 local ERM
sh scripts/launch.sh sup resnet50-BN 10 3 local ERM
sh scripts/launch.sh unsup resnet50-BN 10 3 local ERM
```

for baselines based on KNN training algorithm

```
sh scripts/launch.sh pretrain resnet50 10 3 local KNN
sh scripts/launch.sh sup resnet50 10 3 local KNN
sh scripts/launch.sh unsup resnet50 10 3 local KNN
```

## License

This source code is released under the MIT license, included [here](LICENSE).

### Citation 
If you find this repo useful, please consider citing: 
```
@misc{zhang2023adanpc,
      title={AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation}, 
      author={Yi-Fan Zhang and Xue Wang and Kexin Jin and Kun Yuan and Zhang Zhang and Liang Wang and Rong Jin and Tieniu Tan},
      year={2023},
      eprint={2304.12566},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```