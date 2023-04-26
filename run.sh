# CUDA_VISIBLE_DEVICES=2 nohup python -m domainbed.scripts.train\
#        --data_dir /data2/yifan.zhang/datasets/DGdata/ \
#        --output_dir /data1/yifan.zhang/DADG/T3A/models \
#        --algorithm ERM\
#        --dataset PACS\
#        --hparams "{\"backbone\": \"resnet50\"}"  > resnet50_ERM.out 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python -m domainbed.scripts.train\
#        --data_dir /data2/yifan.zhang/datasets/DGdata/ \
#        --output_dir /data1/yifan.zhang/DADG/T3A/models \
#        --algorithm ERM\
#        --dataset PACS\
#        --hparams "{\"backbone\": \"resnet18\"}"  > resnet18_ERM.out 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -m domainbed.scripts.train\
#        --data_dir /data2/yifan.zhang/datasets/DGdata/ \
#        --output_dir /data1/yifan.zhang/DADG/T3A/models \
#        --algorithm ERM\
#        --dataset PACS\
#        --hparams "{\"backbone\": \"ViT-B16\"}"  > ViT-B16_ERM.out 2>&1 &
nohup python sweep.py > HViT-AdaNPC.out 2>&1 &
sh scripts/launch.sh pretrain HViT 10 1 local ERM
CUDA_VISIBLE_DEVICES=3 nohup python unsupervised_adaptation.py --adapt_algorithm DRM > DRM_individual_drm.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python unsupervised_adaptation.py --adapt_algorithm DRMFull > DRMFull_individual_drm.out 2>&1 &
python -m domainbed.scripts.collect_results  --input_dir=sweep/DRM/ViT-B16