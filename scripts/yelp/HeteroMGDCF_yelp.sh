cd ../../
python -u cf_task_handle.py  \
 HeteroMGDCF --dataset light_gcn_yelp \
        --emb_size 64 \
        --lr 1e-2 \
        --lr_decay 0.98 \
        --z_l2_coef 1e-4 \
        --num_negs 300 \
        --batch_size 8000 \
        --num_epochs 300 \
        --adj_drop_rate 0.97 \
        --alpha 0.1 \
        --beta 0.9 \
        --num_iter 4 \
        --x_drop_rate 0.3 \
        --z_drop_rate 0.1 \
        --edge_drop_rate 0.5 \
        --output_dir results