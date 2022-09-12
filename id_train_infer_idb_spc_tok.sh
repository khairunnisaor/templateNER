for N in 5 15 20 80 90 ; do
    CUDA_VISIBLE_DEVICES=0 python id_train_indobart_spc_tok.py ./exp/template_indobart_idner2k_spc_tok_seed_$N ./outputs/id_best_model_indobart_idner2k_spc_tok_seed_$N $N
    CUDA_VISIBLE_DEVICES=0 python id_inference_indobart_spc_tok.py ./outputs/id_best_model_indobart_idner2k_spc_tok_seed_$N ./id_pred_indobart_idner2k_spc_tok_seed_$N.txt
done

# nohup bash id_train_inf_idb_spc_tok.sh > id_train_infer_indobart_idner2k_spc_tok_seeds_515208090.out 