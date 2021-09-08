python -m scripts.train \
  --dataset_name 'selection' \
  --delim tab \
  --d_type 'local' \
  --pred_len 12 \
  --noise_type gaussian \
  --noise_mix_type global \
  --l2_loss_weight 1 \
  --batch_norm 0 \
  --dropout 0 \
  --batch_size 64 \
  --g_steps 1 \
  --d_steps 2 \
  --checkpoint_every 1 \
  --num_epochs 1000 \
  --pooling_type 'pool_net' \
  --clipping_threshold_g 1.5 \
  --best_k 10 \
  --gpu_num 1 \
  --checkpoint_name gan_test \
  --timing 1 \
  --raw_dataset_folder raw_data \
  --step 5 \
  --tf_emb_dim 256 \
  --tf_ff_size 1024 \
  --pool_emb_dim 256 \
  --bottleneck_dim 128 \
  --mlp_dim 64\
  --noise_dim 8 \
  --layer_count 4 \
  --g_learning_rate 1e-3 \
  --d_learning_rate 1e-3 \
  --heads 8
