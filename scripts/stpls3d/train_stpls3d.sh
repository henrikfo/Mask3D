#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=12.5
CURR_TOPK=200
CURR_QUERY=160
CURR_SIZE=54
CURR_THRESHOLD=0.01

python main_instance_segmentation.py \
general.experiment_name="my_training_5" \
general.project_name="stpls3d" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.5 \
data.num_workers=4 \
data.cache_data=true \
data.cropping_v1=false \
trainer.max_epochs=300 \
general.reps_per_epoch=100 \
data.batch_size=3 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
data.train_mode=train \
