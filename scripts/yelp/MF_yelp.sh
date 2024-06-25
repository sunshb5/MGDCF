cd ../../
python -u cf_task_handle.py  \
  MF --dataset light_gcn_yelp \
        --emb_size 64 \
        --lr 5e-3 \
        --lr_decay 0.995 \
        --z_l2_coef 1e-4 \
        --num_negs 1 \
        --batch_size 8000 \
        --num_epochs 300 \
        --adj_drop_rate 0.97 \
        --alpha 0.1 \
        --num_iter 2 \
        --x_drop_rate 0.0 \
        --z_drop_rate 0.3 \
        --edge_drop_rate 0.5 \
        --output_dir results