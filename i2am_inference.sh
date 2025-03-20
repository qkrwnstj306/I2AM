#### paired
CUDA_VISIBLE_DEVICES=0 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --save_dir ./results \
 --data_root_dir ../VITON/dataset/zalando-hd-resized \
 --generated_image True \
 --per_time_step False \
 --per_attention_head False \
 --controlnet False \
 --new_attention_map False

#  #### unpaired
CUDA_VISIBLE_DEVICES=1 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --unpair \
 --data_root_dir ../VITON/dataset/zalando-hd-resized \
 --save_dir ./results \
  --generated_image True \
 --per_time_step False \
 --per_attention_head False \
 --controlnet False \
 --new_attention_map False