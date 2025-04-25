GPU=$1
DATA=$2
IPC=$3


CUDA_VISIBLE_DEVICES=${GPU} python main_method.py \
--method Tucker \
--vae_model 2DVAE \
--select_mode DAPS \
--dataset ${DATA} \
--eval_mode SS \
--ipc ${IPC} \
--num_eval 5 \
--epoch_eval_train 500 \
--init real \
--lr_net 0.01 \
--model ConvNet3D \
--num_workers 4 \
--random_state 42 \
--compress_ratio 0.75 \
--num_clusters 4 \
--encode_batch_size 8 \
--frames 16 \
--preload \
