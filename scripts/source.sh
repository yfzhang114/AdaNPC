#!/bin/sh
nvidia-smi

nohup python -m domainbed.scripts.sweep delete_incomplete\
       --data_dir=/data2/yifan.zhang/datasets/DGdata/ \
       --output_dir=./sweep/$6/$1 \
       --command_launcher $5\
       --algorithms $6\
       --datasets $2\
       --n_hparams 1\
       --n_trials_from $3 \
       --n_trials $4 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation > lanuch_delete.out 2>&1 &

python -m domainbed.scripts.sweep launch\
       --data_dir=/data2/yifan.zhang/datasets/DGdata/ \
       --output_dir=./sweep/$6/$1 \
       --command_launcher $5\
       --algorithms $6\
       --datasets $2\
       --n_hparams 1\
       --n_trials_from $3 \
       --n_trials $4 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation > lanuch_pretrain_$6_$1.out 2>&1 &
