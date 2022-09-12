for N in 5 10 25 50 75 ; do
    # 600
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_600_5.csv ./exp/template_indobart_600_5_seed_$N ./outputs/id_best_model_indobart_600_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_600_5_seed_$N ./id_pred_indobart_600_5_seed_$N.txt
    # 700
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_700_5.csv ./exp/template_indobart_700_5_seed_$N ./outputs/id_best_model_indobart_700_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_700_5_seed_$N ./id_pred_indobart_700_5_seed_$N.txt
    # 800
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_800_5.csv ./exp/template_indobart_800_5_seed_$N ./outputs/id_best_model_indobart_800_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_800_5_seed_$N ./id_pred_indobart_800_5_seed_$N.txt
    # 900
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_900_5.csv ./exp/template_indobart_900_5_seed_$N ./outputs/id_best_model_indobart_900_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_900_5_seed_$N ./id_pred_indobart_900_5_seed_$N.txt
    # 1000
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_1000_5.csv ./exp/template_indobart_1000_5_seed_$N ./outputs/id_best_model_indobart_1000_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_1000_5_seed_$N ./id_pred_indobart_1000_5_seed_$N.txt
    # 1100
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_1100_5.csv ./exp/template_indobart_1100_5_seed_$N ./outputs/id_best_model_indobart_1100_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_1100_5_seed_$N ./id_pred_indobart_1100_5_seed_$N.txt
    # 1200
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_1200_5.csv ./exp/template_indobart_1200_5_seed_$N ./outputs/id_best_model_indobart_1200_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_1200_5_seed_$N ./id_pred_indobart_1200_5_seed_$N.txt
    # 1300
    CUDA_VISIBLE_DEVICES=2 python id_train_indobart1.py ./data/idner2k-fewshot/train_id_1300_5.csv ./exp/template_indobart_1300_5_seed_$N ./outputs/id_best_model_indobart_1300_5_seed_$N $N
    CUDA_VISIBLE_DEVICES=2 python id_inference_indobart1.py ./outputs/id_best_model_indobart_1300_5_seed_$N ./id_pred_indobart_1300_5_seed_$N.txt
done

# nohup bash id_train_inf_idb.sh > id_train_infer_indobart_600-1300_5_seed.out 