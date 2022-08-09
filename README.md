# Predictive Ensemble Learning Based on the Dynamic Predictor for Concept Drift Scenarios
NTU MSLAB Concept drift team

## File description
* experiment_batch0.py: Fine-tuning  
* experiment_dynse.py: Dynse framework
* experiment_dtel.py: DTEL framework
* experiment_ddcw.py: DDCW framework
* experiment_ddgda.py: DDG-DA framework
* experiment_dp_future.py: DP.FUTURE framework
* experiment_dp_all.py: DP.ALL framework
* datasets.py: Datasets for the experiment  
* utils.py: Some utilities of training and testing methods  
* models.py: Models for the experiment, e.g., classifier and dynamic predictor 

## Environment
python 3.8

## Package
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

## How to execute codes
* If you want to run all the experiments with seed=0 (will take a few days):
```
sh run_command.sh
```
* You can also apply a method on a dataset (the Hyperparameters Section shows the detailed arguments of each method). For example, if you want to run DP.FUTURE on the Rotation dataset:
```
python3 experiment_dp_future.py --seed 0 --finetuned_epochs 50 --activate_dynamic_t 3 --max_ensemble_size 3 --dataset rotate --classifier lr --device cuda:0 --voting soft --mask_old_classifier
```

## Hyperparameters
* experiment_batch0.py
```
--seed: set random seed (type=int, default=0)
--last_step_method: set the last step method. It must be set to 'none' if Fine-tuning is performed (type=str, default='none', option={'none', 'soft', 'hard', 'cost'})
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction. It can be ignored if Fine-tuning is performed (type=int, default=3)
--time_window: set the maximum input vector sequence length of the dynamic predictor. It can be ignored if Fine-tuning is performed (type=int, default=3)
--dataset: set the dataset for the experiment. Please refer to dataset_dict in datasets.py for more information (type=str, default='translate', option={'translate', 'rotate', 'ellipse', 'progress', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting into a training set and a validation set (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
```

* experiment_dynse.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--dataset: set the dataset for the experiment. Please refer to dataset_dict in datasets.py for more information (type=str, default='translate', option={'translate', 'rotate', 'ellipse', 'progress', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting into a training set and a validation set (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=3)
--max_validation_window_size: maximum number of data batches to store in the accuracy estimation window (type=int, default=1)
--neighbor_size: k in the k-nearest neighbors algorithm (type=int, default=5)
```

* experiment_dtel.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--dataset: set the dataset for the experiment. Please refer to dataset_dict in datasets.py for more information (type=str, default='translate', option={'translate', 'rotate', 'ellipse', 'progress', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting into a training set and a validation set (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=3)
--finetuned_epochs: number of epochs to train for fine-tuning the historical classifiers (type=int, default=50)
--epsilon: add epsilon to the denominator to avoid being divided by zero when calculating the diversity value (type=float, default=1e-5)
```

* experiment_ddcw.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--dataset: set the dataset for the experiment. Please refer to dataset_dict in datasets.py for more information (type=str, default='translate', option={'translate', 'rotate', 'ellipse', 'progress', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting into a training set and a validation set (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=3)
--beta: set beta parameter. It should be larger than 1 (type=float, default=1.1)
--life_time_coefficient: set life-time coefficient. It should be smaller or equal than 1 (type=float, default=0.9)
--epsilon: add epsilon to the denominator to avoid being divided by zero when calculating the diversity value (type=float, default=1e-5)
```

* experiment_ddgda.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--Q_lr: set predictor learning rate (type=float, default=1e-3)
--Q_epochs: set predictor training epochs (type=int, default=50)
--Q_decay: set predictor weight decay (type=float, default=0)
--activate_dynamic_t: set when to activate the predictor for prediction (type=int, default=3)
--time_window: set the maximum number of data batches to consider when training the predictor. (type=int, default=3)
--dataset: set the dataset for the experiment. Please refer to dataset_dict in datasets.py for more information (type=str, default='translate', option={'translate', 'rotate', 'ellipse', 'progress', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting into a training set and a validation set (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--finetune_classifier_method: set the training method for the classifier (type=str, default='soft', option={'soft', 'hard'})
```

* experiment_dp_future.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--finetuned_epochs: number of epochs to train for fine-tuning the historical classifiers (type=int, default=50)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=3)
--dataset: set the dataset for the experiment. Please refer to dataset_dict in datasets.py for more information (type=str, default='translate', option={'translate', 'rotate', 'ellipse', 'progress', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting into a training set and a validation set (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--life_time_coefficient: set life-time coefficient. It should be smaller or equal than 1 (type=float, default=1.0)
--alpha: set the tradeoff between current feedback and previous weights (type=float, default=0.5)
--voting: set the voting mechanism during prediction (type=str, default='soft', option={'soft', 'hard'})
--mask_old_classifier: whether to exclude old classifiers during prediction. It must be set if DP.FUTURE is performed (action="store_true")
```

* experiment_dp_all.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--finetuned_epochs: number of epochs to train for fine-tuning the historical classifiers (type=int, default=50)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=3)
--dataset: set the dataset for the experiment. Please refer to dataset_dict in datasets.py for more information (type=str, default='translate', option={'translate', 'rotate', 'ellipse', 'progress', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting into a training set and a validation set (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--life_time_coefficient: set life-time coefficient. It should be smaller or equal than 1 (type=float, default=1.0)
--alpha: set the tradeoff between current feedback and previous weights (type=float, default=0.5)
--voting: set the voting mechanism during prediction (type=str, default='soft', option={'soft', 'hard'})
--mask_old_classifier: whether to exclude old classifiers during prediction. It must "not" be set if DP.ALL is performed (action="store_true")
```

## Note
* If --last_step_method is set to 'none' in experiment_batch0.py, then the whole algorithm is equivalent to finetuning, and --activate_dynamic_t has no effect because the dynamic predictor won't be used for prediction
* For more information about the options of --dataset, please see dataset_dict in datasets.py
* For Dynse framework, please refer to https://www.researchgate.net/publication/323727593_Adapting_the_Dynamic_Classifier_Selection_for_Concept_Drift_Scenarios
* For DTEL framework, please refer to https://ieeexplore.ieee.org/document/8246541
* For DDCW framework, please refer to https://ieeexplore.ieee.org/document/9378625
* For DDG-DA framework, please refer to https://arxiv.org/abs/2201.04038
hi my name is
Austin 
good  
ok