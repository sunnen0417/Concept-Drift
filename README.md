# Concept-Drift
NTU MSLAB Concept drift team

## File description
* experiment_batch0.py: Train the classifier with only data batch t, train the DP with only data batch 0
* experiment_only_all.py: Train the classifier with only data batch t, train the DP with all data batches
* experiment_all_all.py: Train the classifier with all data batches, train the DP with all data batches
* experiment_vae.py: Train the classifier with VAE samples and data batch t, train the DP with VAE samples and data batch t  
* experiment_dynse.py: Dynse framework
* experiment_dtel.py: DTEL framework
* experiment_ddcw.py: DDCW framework
* experiment_test_dp.py: Train the dynamic predictor using testing data
* experiment_test_multi_dp.py: Train multiple dynamic predictors using bootstrapping on testing data
* experiment_dtel_dp.py: Use the dynamic predictor to generate a pool of future classifiers based on validation set. Average soft voting is done by previous classifiers and future classifiers 
* experiment_test_dp_dtel.py: Use the dynamic predictor to generate a pool of future classifiers based on testing set. Average soft voting is done by previous classifiers and future classifiers
* experiment_test_dp_dtel_rule.py: Use the dynamic predictor to generate a pool of future classifiers based on testing set. Ruled-based weighted voting is done by previous classifiers and future classifiers
* datasets.py: Datasets for the experiment  
* utils.py: Some utilities of training and testing methods  
* models.py: Models for the experiment, e.g., classifier, DP and VAE  
* plot_result.py: Draw acc. curves  
* visualize.py: PCA visualization on a dataset    

## How to execute codes
* For example, if you want to run experiment_vae.py on gas sensor dataset with soft label training as the last step method (others with default values):
```
python3 experiment_vae.py --last_step_method soft --dataset gas
```

## Hyperparameters
* experiment_batch0.py
```
--seed: set random seed (type=int, default=0)
--last_step_method: set the last step method (type=str, default='none', option={'none', 'soft', 'hard', 'cost'})
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--time_window: set the maximum input vector sequence length of the dynamic predictor (type=int, default=3)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
```

* experiment_only_all.py
```
--seed: set random seed (type=int, default=0)
--last_step_method: set the last step method (type=str, default='none', option={'none', 'soft', 'hard', 'cost'})
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--time_window: set the maximum input vector sequence length of the dynamic predictor (type=int, default=3)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--ckpt_dir: the directory to store checkpoints of the history classifier (type=str, default='./ckpt_only')
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
```

* experiment_all_all.py
```
--seed: set random seed (type=int, default=0)
--last_step_method: set the last step method (type=str, default='none', option={'none', 'soft', 'hard', 'cost'})
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--time_window: set the maximum input vector sequence length of the dynamic predictor (type=int, default=3)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--ckpt_dir: the directory to store checkpoints of the history classifier (type=str, default='./ckpt_all')
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
```

* experiment_vae.py
```
--seed: set random seed (type=int, default=0)
--last_step_method: set the last step method (type=str, default='none', option={'none', 'soft', 'hard', 'cost'})
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--vae_lr: set vae learning rate (type=float, default=1e-3)
--vae_epochs: set maximum vae training epochs (type=int, default=250)
--vae_decay: set vae weight decay (type=float, default=0)
--theta: set the threshold for the vae wrap function to cover samples. Should be less than 0.5 (type=float, default=0.1)
--sample_n: set number of data to sample per vae (class) (type=int, default=200)
--eps: set ball radius to uncover a sample (type=float, default=0.1)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--time_window: set the maximum input vector sequence length of the dynamic predictor (type=int, default=3)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
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
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--max_pool_size: maximum number of base classifiers in the ensemble (type=int, default=25)
--max_validation_window_size: maximum number of data batches to store in the validation set (type=int, default=4)
--neighbor_size: k in k-nearest neighbors algorithm (type=int, default=5)
```

* experiment_dtel.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--num_classifiers: maximum number of base classifiers in the ensemble (type=int, default=3)
--finetuned_epochs: number of epochs to train for fine-tuning the historical classifiers (type=int, default=1)
--epsilon: add epsilon to the denominator to avoid divided by zero when calculating the diversity value (type=float, default=1e-5)
```

* experiment_ddcw.py
```
--seed: set random seed (type=int, default=0)
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=3)
--beta: set beta parameter. Should be larger than 1 (type=float, default=1.1)
--life_time_coefficient: set life time coefficient. Should be smaller than 1 (type=float, default=0.9)
--epsilon: add epsilon to the denominator to avoid divided by zero when calculating the diversity value (type=float, default=1e-5)
```

* experiment_test_dp.py
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
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=25)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
```

* experiment_test_multi_dp.py
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
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=25)
--max_dp_size: maximum number of dynamic predictors (type=int, default=3)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
```

* experiment_dtel_dp.py
```
--seed: set random seed (type=int, default=0)
--last_step_method: set the last step method (type=str, default='soft', option={'none', 'soft', 'hard', 'cost'})
--num_workers: set workers number (type=int, default=0)
--batch_size: set batch size (type=int, default=64)
--lr: set classifier learning rate (type=float, default=2e-3)
--epochs: set classifier training epochs (type=int, default=50)
--decay: set classifier weight decay (type=float, default=5e-5)
--d_lr: set dynamic predictor learning rate (type=float, default=1e-3)
--d_epochs: set dynamic predictor training epochs (type=int, default=50)
--d_decay: set dynamic predictor weight decay (type=float, default=0)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--max_pool_size: maximum number of base classifiers in the ensemble (type=int, default=3)
--max_validation_window_size: maximum number of data batches to store in the validation set (type=int, default=3)
--neighbor_size: k in k-nearest neighbors algorithm (type=int, default=5)
--finetuned_epochs: number of epochs to train for fine-tuning the historical classifiers (type=int, default=5)
--cluster_assumption: set whether to activate the cluster assumption (action="store_true")
--ca_epochs: set cluster assumption fine-tuning epochs (type=int, default=10)
--ca_lr: set cluster assumption learning rate (type=float, default=0.0005)
--ema_decay: set cluster assumption model weight decay (type=float, default=0.998)
--perturb_radius: set perturbation radius for virtual adversarial perturbation (type=float, default=0.1)
--XI: set scaling size for virtual adversarial perturbation (type=float, default=0.1)
```

* experiment_test_dp_dtel.py
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
--finetuned_epochs: number of epochs to train for fine-tuning the historical classifiers (type=int, default=5)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=25)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
```

* experiment_test_dp_dtel_rule.py
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
--finetuned_epochs: number of epochs to train for fine-tuning the historical classifiers (type=int, default=5)
--activate_dynamic_t: set when to activate the dynamic predictor for prediction (type=int, default=3)
--max_ensemble_size: maximum number of base classifiers in the ensemble (type=int, default=25)
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype', ...})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
--train_ratio: set the ratio of the training set while splitting (type=float, default=0.8)
--patience: set the patience for early stopping (type=int, default=7)
--life_time_coefficient: set life time coefficient (type=float, default=1.0)
--alpha: set the tradeoff between current feedback and previous weights (type=float, default=0.5)
--voting: set the voting mechanism during prediction (type=str, default='soft', option={'soft', 'hard'})
--mask_old_classifier: whether to exclude old classifiers during prediction (action="store_true")
```

## Note
* For --theta, --sample_n and --eps in experiment_vae.py, please refer to https://link.springer.com/article/10.1007%2Fs00521-021-06154-9
* If --last_step_method is set to 'none', then the whole algorithm is equivalent to finetuning, and --activate_dynamic_t has no effect because the dynamic predictor won't be used for prediction
* For Dynse framework, please refer to https://www.researchgate.net/publication/323727593_Adapting_the_Dynamic_Classifier_Selection_for_Concept_Drift_Scenarios
* For DTEL framework, please refer to https://ieeexplore.ieee.org/document/8246541
* For DDCW framework, please refer to https://ieeexplore.ieee.org/document/9378625
* For more information about the options of --dataset, please see dataset_dict in datasets.py