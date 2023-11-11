#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=12.5
CURR_TOPK=200
CURR_QUERY=160
CURR_SIZE=54
CURR_THRESHOLD=0.01

python3.10 main_instance_segmentation.py \
general.experiment_name="my_training_05" \
general.project_name="valid" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.5 \
data.num_workers=4 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=1 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="saved/my_training_5/last-epoch.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK} \
general.export=false \
general.save_visualizations=true
