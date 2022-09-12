for N in 10 25 50 75 100 ; do
    CUDA_VISIBLE_DEVICES=0 python id_train_mbart_spc_tok.py ./exp/template_mbart_idner2k_spc_tok_seed_$N ./outputs/id_best_model_mbart_idner2k_spc_tok_seed_$N $N
    CUDA_VISIBLE_DEVICES=0 python id_inference_mbart_spc_tok.py ./outputs/id_best_model_mbart_idner2k_spc_tok_seed_$N ./id_pred_mbart_idner2k_spc_tok_seed_$N.txt
done

# nohup bash id_train_inf_mbart_spc_tok.sh > id_train_mbart_idner2k_spc_tok_seed_10255075100.out 