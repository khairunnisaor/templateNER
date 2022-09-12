for N in 5 10 25 ; do    
    # 900
    CUDA_VISIBLE_DEVICES=0 python id_train_mbart1.py ./data/idner2k-fewshot/train_id_900_1.csv ./exp/template_mbart_900_1_seed_$N ./outputs/id_best_model_mbart_900_1_seed_$N $N
    CUDA_VISIBLE_DEVICES=0 python id_inference_mbart2.py ./outputs/id_best_model_mbart_900_1_seed_$N ./id_pred_mbart_900_1_seed_$N.txt
done

for N in 5 10 25 ; do   
    # 1000
    CUDA_VISIBLE_DEVICES=0 python id_train_mbart1.py ./data/idner2k-fewshot/train_id_1000_1.csv ./exp/template_mbart_1000_1_seed_$N ./outputs/id_best_model_mbart_1000_1_seed_$N $N
    CUDA_VISIBLE_DEVICES=0 python id_inference_mbart2.py ./outputs/id_best_model_mbart_1000_1_seed_$N ./id_pred_mbart_1000_1_seed_$N.txt
done

for N in 5 10 25 ; do   
    # 1100
    CUDA_VISIBLE_DEVICES=0 python id_train_mbart1.py ./data/idner2k-fewshot/train_id_1100_1.csv ./exp/template_mbart_1100_1_seed_$N ./outputs/id_best_model_mbart_1100_1_seed_$N $N
    CUDA_VISIBLE_DEVICES=0 python id_inference_mbart2.py ./outputs/id_best_model_mbart_1100_1_seed_$N ./id_pred_mbart_1100_1_seed_$N.txt
done

for N in 5 10 25 ; do   
    # 1200
    CUDA_VISIBLE_DEVICES=0 python id_train_mbart1.py ./data/idner2k-fewshot/train_id_1200_1.csv ./exp/template_mbart_1200_1_seed_$N ./outputs/id_best_model_mbart_1200_1_seed_$N $N
    CUDA_VISIBLE_DEVICES=0 python id_inference_mbart2.py ./outputs/id_best_model_mbart_1200_1_seed_$N ./id_pred_mbart_1200_1_seed_$N.txt
done

# for N in 5 10 25 ; do   
#     # 800
#     CUDA_VISIBLE_DEVICES=0 python id_train_mbart1.py ./data/idner2k-fewshot/train_id_800_1.csv ./exp/template_mbart_800_1_seed_$N ./outputs/id_best_model_mbart_800_1_seed_$N $N
#     CUDA_VISIBLE_DEVICES=0 python id_inference_mbart2.py ./outputs/id_best_model_mbart_800_1_seed_$N ./id_pred_mbart_800_1_seed_$N.txt
# done

# nohup bash id_train_inf_mbart.sh > id_train_inf_mbart_fewshot_1_seed_2.out 