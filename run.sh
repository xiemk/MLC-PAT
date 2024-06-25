

data_name=vg256
data_dir= xxx
device_id=0,1,2,3
# # # ===========================

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_PAT-T.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name resnet101 --pretrain_type in1k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross --print_freq 200 --early_stop

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_PAT-T.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl_v1 --pretrain_type in1k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross --print_freq 200 --early_stop

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_PAT-T.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl_v2 --pretrain_type in21k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross --print_freq 200 --early_stop

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_PAT-T  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl --pretrain_type in1k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross --print_freq 200 --early_stop

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_PAT-T  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2 --pretrain_type in21k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross --print_freq 200 --early_stop

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_PAT-T  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2 --pretrain_type oi \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross --print_freq 200 --early_stop


######################################################################################################


# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_inference_dist.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name resnet101 --pretrain_type in1k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross 

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_inference_dist.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl_v1 --pretrain_type in1k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross 

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_inference_dist.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl_v2 --pretrain_type in21k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross 

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_inference_dist.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl --pretrain_type in1k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross 

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_inference_dist.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2 --pretrain_type in21k \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross 

# CUDA_VISIBLE_DEVICES=$device_id python -m torch.distributed.launch --nproc_per_node 4 run_inference_dist.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2 --pretrain_type oi \
# --batch_size 128 --image_size 448 --distributed \
# --logits_attention cross 

#################################################################################################################



# CUDA_VISIBLE_DEVICES=$device_id python run_inference.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name resnet101  --pretrain_type in1k \
# --batch_size 128 --image_size 448 


# CUDA_VISIBLE_DEVICES=$device_id python run_inference.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl  --pretrain_type in1k \
# --batch_size 128 --image_size 448 

# CUDA_VISIBLE_DEVICES=$device_id python run_inference.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl_v2  --pretrain_type in21k \
# --batch_size 128 --image_size 448 

# CUDA_VISIBLE_DEVICES=$device_id python run_inference.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl  --pretrain_type in1k \
# --batch_size 128 --image_size 448 

# CUDA_VISIBLE_DEVICES=$device_id python run_inference.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2  --pretrain_type in21k \
# --batch_size 128 --image_size 448  

# CUDA_VISIBLE_DEVICES=$device_id python run_inference.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2  --pretrain_type oi \
# --batch_size 128 --image_size 448  

##########################################################################

# CUDA_VISIBLE_DEVICES=$device_id python run_PAT-I.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl --pretrain_type in1k \
# --batch_size 128 --image_size 448 

# CUDA_VISIBLE_DEVICES=$device_id python run_PAT-I.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name tresnetl_v2 --pretrain_type in21k \
# --batch_size 128 --image_size 448 

# CUDA_VISIBLE_DEVICES=$device_id python run_PAT-I.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl --pretrain_type in1k \
# --batch_size 64 --image_size 448 

# CUDA_VISIBLE_DEVICES=$device_id python run_PAT-I.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2 --pretrain_type in21k \
# --batch_size 64 --image_size 448 

# CUDA_VISIBLE_DEVICES=$device_id python run_PAT-I.py  \
# --data_name $data_name  --data_dir $data_dir  \
# --model_name q2l_tresnetl_v2 --pretrain_type oi \
# --batch_size 64 --image_size 448 