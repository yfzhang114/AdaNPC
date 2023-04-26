#!/bin/sh



python -m domainbed.scripts.sweep unsupervised_adaptation\
       --data_dir=./data2/yifan.zhang/datasets/DGdata/ \
       --output_dir=./sweep/$1 \
       --command_launcher multi_gpu\
       --algorithms $6\
       --datasets $2\
       --n_hparams 1\
       --n_trials_from $3 \
       --n_trials $4 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation
