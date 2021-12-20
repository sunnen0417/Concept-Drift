# Concept-Drift
NTU MSLAB Concept drift team

## File description
* experiment_batch0.py: Train the classifier with only data batch t, train the DP with only data batch 0  
* experiment_vae.py: Train the classifier with VAE samples and data batch t, train the DP with VAE samples and data batch t  
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
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype'})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
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
--dataset: set the dataset for the experiment (type=str, default='translate', option={'translate', 'rotate', 'ball', 'gas', 'covertype'})
--classifier: set the classifier type (type=str, default='lr', option={'lr':logistic regression, 'mlp':neural network})
--device: set device (type=str, default='cuda:0', option={'cpu', 'cuda:0', 'cuda:1', ...})
```

## Note
* For --theta, --sample_n and --eps in experiment_vae.py, please refer to https://link.springer.com/article/10.1007%2Fs00521-021-06154-9
* If --last_step_method is set to 'none', then the whole algorithm is equivalent to finetuning, and --activate_dynamic_t, --time_window have no effect because the dynamic predictor won't be used for both training and prediction