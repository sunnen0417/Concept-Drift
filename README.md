# Concept-Drift

- NTU MSLAB Concept drift team

## Settings

- Use the settings on 2021/11/25 slide

- move with Normal distribution step with mean = $\mu$, variance = $0.5\mu$ => $N(\mu, 0.5\mu)$

- THs = 10 percentile, THe = 90 percentile

- Time point = k (with k + 1 threshold)

## Hyperparameters

```
-seed: set random seed (type=int default=0)
-num_workers: set workers number (type=int default=0)
-batch_size: set batch size (type=int default=64)
-lr: set MLP learning rate (type=float default=2e-3)
-epochs: set MLP epochs (type=int default=50)
-decay: set MLP decay (type=float default=5e-5)
-d_lr: set Dynamic Predictor learning rate (type=float default=1e-3)
-d_epochs: set Dynamic Predictor epochs (type=int default=50)
-d_decay: set Dynamic Predictor decay (type=float default=0)
-method: set method 'finetune', 'soft', 'hard', 'cost' (type=str default='finetune')
-activate_dynamic_t: set activate Dynamic Predictor time (type=int default=3)
-time_window: set time window (type=int default=3)
-time_slice: set time slice (type=int default=21)
-class_num: set number of class eg: 2 = binary classification (type=int default=2)
```

## How to use

```
python3 main.py -method finetune -time_slice 21 -time_window 3
```

## Where to find logs

- You will find it at
```
logs/{method}_ts{time_slice}_tw{time_window}.log
```

## Algorithm

- from slide 2021/10/28

```
n: number of data batches; w: maximum window size
train data_0
get softmax_0 on data_0 
for t = 1 to n-1 do
	test data_t
	train data_t
	get softmax_t on data_0
	train dynamic predictor {softmax_(t-w),...,softmax_(t-2), softmax_(t-1)} --> softmax_t
	predict softmax_(t+1) on data_0 {softmax_(t-w+1),...,softmax_(t-1), softmax_(t)} --> predicted_softmax_(t+1)
	predict decision boundary t+1 by training data_0 --> predicted_softmax_(t+1)
end for
```
