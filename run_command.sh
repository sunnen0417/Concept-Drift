# 1. Synthetic dataset
# Translate
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset translate --classifier lr --device cuda:0 > translate_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset translate --classifier lr --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > translate_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset translate --classifier lr --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > translate_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset translate --classifier lr --device cuda:0 --max_ensemble_size 3 > translate_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset translate --classifier lr --device cuda:0 --finetune_classifier_method soft > translate_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset translate --classifier lr --device cuda:0 --voting soft --mask_old_classifier > translate_dp_future_0.txt

# Rotate
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset rotate --classifier lr --device cuda:0 > rotate_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset rotate --classifier lr --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > rotate_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset rotate --classifier lr --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > rotate_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset rotate --classifier lr --device cuda:0 --max_ensemble_size 3 > rotate_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset rotate --classifier lr --device cuda:0 --finetune_classifier_method soft > rotate_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset rotate --classifier lr --device cuda:0 --voting soft --mask_old_classifier > rotate_dp_future_0.txt

# Ellipse
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset ellipse --classifier mlp --device cuda:0 > ellipse_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset ellipse --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > ellipse_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset ellipse --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > ellipse_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset ellipse --classifier mlp --device cuda:0 --max_ensemble_size 3 > ellipse_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset ellipse --classifier mlp --device cuda:0 --finetune_classifier_method soft > ellipse_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset ellipse --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > ellipse_dp_future_0.txt

# Progress
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --batch_size 32 --dataset progress --classifier mlp --device cuda:0 > progress_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --batch_size 32 --dataset progress --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > progress_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --batch_size 32 --dataset progress --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > progress_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --batch_size 32 --dataset progress --classifier mlp --device cuda:0 --max_ensemble_size 3 > progress_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --batch_size 32 --activate_dynamic_t 3 --time_window 3 --dataset progress --classifier mlp --device cuda:0 --finetune_classifier_method soft > progress_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --batch_size 32 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset progress --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > progress_dp_future_0.txt

# 2. Regression dataset + threshold
# Graduate Admission
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset admission --classifier mlp --device cuda:0 > admission_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset admission --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > admission_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset admission --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > admission_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset admission --classifier mlp --device cuda:0 --max_ensemble_size 3 > admission_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset admission --classifier mlp --device cuda:0 --finetune_classifier_method soft > admission_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset admission --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > admission_dp_future_0.txt

# House Sale
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset sale --classifier mlp --device cuda:0 > sale_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset sale --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > sale_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset sale --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > sale_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset sale --classifier mlp --device cuda:0 --max_ensemble_size 3 > sale_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset sale --classifier mlp --device cuda:0 --finetune_classifier_method soft > sale_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset sale --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > sale_dp_future_0.txt

# Wine Red
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset wine_red --classifier mlp --device cuda:0 > wine_red_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset wine_red --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > wine_red_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset wine_red --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > wine_red_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset wine_red --classifier mlp --device cuda:0 --max_ensemble_size 3 > wine_red_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset wine_red --classifier mlp --device cuda:0 --finetune_classifier_method soft > wine_red_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset wine_red --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > wine_red_dp_future_0.txt

# Wine White
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset wine_white --classifier mlp --device cuda:0 > wine_white_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset wine_white --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > wine_white_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset wine_white --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > wine_white_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset wine_white --classifier mlp --device cuda:0 --max_ensemble_size 3 > wine_white_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset wine_white --classifier mlp --device cuda:0 --finetune_classifier_method soft > wine_white_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset wine_white --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > wine_white_dp_future_0.txt

# Power Plant
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset power --classifier mlp --device cuda:0 > power_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset power --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > power_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset power --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > power_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset power --classifier mlp --device cuda:0 --max_ensemble_size 3 > power_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset power --classifier mlp --device cuda:0 --finetune_classifier_method soft > power_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset power --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > power_dp_future_0.txt

# California Housing
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset house --classifier mlp --device cuda:0 > house_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset house --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > house_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset house --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > house_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset house --classifier mlp --device cuda:0 --max_ensemble_size 3 > house_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset house --classifier mlp --device cuda:0 --finetune_classifier_method soft > house_ddgda_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset house --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > house_dp_future_0.txt

# Ford Price
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset price --classifier mlp --device cuda:0 > price_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset price --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > price_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset price --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > price_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset price --classifier mlp --device cuda:0 --max_ensemble_size 3 > price_ddcw_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset price --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > price_dp_future_0.txt

# 3. Real world dataset
# Gas Sensor Array
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset gas --classifier mlp --device cuda:0 > gas_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset gas --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > gas_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset gas --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > gas_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset gas --classifier mlp --device cuda:0 --max_ensemble_size 3 > gas_ddcw_0.txt
## DP.ALL
python3 experiment_dp_all.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset gas --classifier mlp --device cuda:0 --voting soft > gas_dp_all_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset gas --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > gas_dp_future_0.txt

# Covertype
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset covertype --classifier mlp --device cuda:0 > covertype_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset covertype --classifier mlp --device cuda:0 --max_ensemble_size 25 --max_validation_window_size 4 > covertype_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset covertype --classifier mlp --device cuda:0 --max_ensemble_size 25 --finetuned_epochs 50 > covertype_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset covertype --classifier mlp --device cuda:0 --max_ensemble_size 25 > covertype_ddcw_0.txt
## DP.ALL
python3 experiment_dp_all.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 25 --dataset covertype --classifier mlp --device cuda:0 --voting soft > covertype_dp_all_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 25 --dataset covertype --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > covertype_dp_future_0.txt

# Electricity
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset electricity --classifier mlp --device cuda:0 > electricity_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset electricity --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > electricity_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset electricity --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > electricity_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset electricity --classifier mlp --device cuda:0 --max_ensemble_size 3 > electricity_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset electricity --classifier mlp --device cuda:0 --finetune_classifier_method soft > electricity_ddgda_0.txt
## DP.ALL
python3 experiment_dp_all.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset electricity --classifier mlp --device cuda:0 --voting soft > electricity_dp_all_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset electricity --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > electricity_dp_future_0.txt

# Airline
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --batch_size 32 --dataset airline --classifier mlp --device cuda:0 > airline_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --batch_size 32 --dataset airline --classifier mlp --device cuda:0 --max_ensemble_size 25 --max_validation_window_size 4 > airline_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --batch_size 32 --dataset airline --classifier mlp --device cuda:0 --max_ensemble_size 25 --finetuned_epochs 50 > airline_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --batch_size 32 --dataset airline --classifier mlp --device cuda:0 --max_ensemble_size 25 > airline_ddcw_0.txt
## DP.ALL
python3 experiment_dp_all.py --seed 0 --batch_size 32 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 25 --dataset airline --classifier mlp --device cuda:0 --voting soft > airline_dp_all_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --batch_size 32 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 25 --dataset airline --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > airline_dp_future_0.txt

# Weather
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset weather --classifier mlp --device cuda:0 > weather_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset weather --classifier mlp --device cuda:0 --max_ensemble_size 25 --max_validation_window_size 4 > weather_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset weather --classifier mlp --device cuda:0 --max_ensemble_size 25 --finetuned_epochs 50 > weather_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset weather --classifier mlp --device cuda:0 --max_ensemble_size 25 > weather_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 25 --dataset weather --classifier mlp --device cuda:0 --finetune_classifier_method soft > weather_ddgda_0.txt
## DP.ALL
python3 experiment_dp_all.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 25 --dataset weather --classifier mlp --device cuda:0 --voting soft > weather_dp_all_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 25 --dataset weather --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > weather_dp_future_0.txt

# Online News Popularity
## Finetune
python3 experiment_batch0.py --seed 0 --last_step_method none --dataset onp --classifier mlp --device cuda:0 > onp_cls_t_dp_0_none_0.txt
## Dynse
python3 experiment_dynse.py --seed 0 --dataset onp --classifier mlp --device cuda:0 --max_ensemble_size 3 --max_validation_window_size 1 > onp_dynse_0.txt
## DTEL
python3 experiment_dtel.py --seed 0 --dataset onp --classifier mlp --device cuda:0 --max_ensemble_size 3 --finetuned_epochs 50 > onp_dtel_0.txt
## DDCW
python3 experiment_ddcw.py --seed 0 --dataset onp --classifier mlp --device cuda:0 --max_ensemble_size 3 > onp_ddcw_0.txt
## DDG-DA
python3 experiment_ddgda.py --seed 0 --activate_dynamic_t 3 --time_window 3 --dataset onp --classifier mlp --device cuda:0 --finetune_classifier_method soft > onp_ddgda_0.txt
## DP.ALL
python3 experiment_dp_all.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset onp --classifier mlp --device cuda:0 --voting soft > onp_dp_all_0.txt
## DP.FUTURE
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset onp --classifier mlp --device cuda:0 --voting soft --mask_old_classifier > onp_dp_future_0.txt
