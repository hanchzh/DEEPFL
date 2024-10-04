## Requirements
python>=3.6  
pytorch>=0.4

## Run



Federated learning with MLP and CNN is produced by:
> python [main_cfed.py](back.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset NOAA --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.

