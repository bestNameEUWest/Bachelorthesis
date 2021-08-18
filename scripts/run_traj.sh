python -m scripts.train \
  --dataset_name 'zara1' \
  --delim tab \
  --d_type 'local' \
  --pred_len 12 \
  --tf_emb_dim 128 \
  --tf_ff_size 128 \
  --pool_emb_dim 16 \
  --bottleneck_dim 128 \
  --mlp_dim 64 \
  --noise_dim 8 \
  --noise_type gaussian \
  --noise_mix_type global \
  --l2_loss_weight 1 \
  --batch_norm 0 \
  --dropout 0 \
  --batch_size 32 \
  --g_learning_rate 1e-3 \
  --g_steps 1 \
  --d_learning_rate 1e-3 \
  --d_steps 2 \
  --checkpoint_every 10 \
  --print_every 50 \
  --num_iterations 20000 \
  --num_epochs 500 \
  --pooling_type 'pool_net' \
  --clipping_threshold_g 1.5 \
  --best_k 10 \
  --gpu_num 1 \
  --checkpoint_name gan_test \
  --restore_from_checkpoint 0
