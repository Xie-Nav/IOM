# Implicit Obstacle Map-driven Indoor Navigation Model for Robust Obstacle Avoidance

## Setup
- Clone the repository `git clone https://github.com/xwaiyy123/object-navigation.git` and move into the top level directory `cd object-navigation`
- Create conda environment. `pip install -r requirements.txt`
## dataset
- Download the [dataset](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view), which refers to [ECCV-VN](https://github.com/xiaobaishu0097/ECCV-VN). 
- You can also use the [DETR object detection features](https://drive.google.com/file/d/1d761VxrwctupzOat4qxsLCm5ndC4wA-M/view?usp=sharing).


## Training and Evaluation

### Train our DAT model
`python main.py --title IOM --model IOM --workers 12 --gpu-ids 0 1 --max-ep 3000000 --log-dir runs/RL_train --save-model-dir trained_models/RL_train --pretrained-trans trained_models/pretrain/checkpoint0004.pth --data-dir /opt/data/private/datasets/AI2Thor_offline_data_2.0.2/` 
### Evaluate our DAT model
`python full_eval.py --title IOM --model IOM --results-json eval_results/IOM.json --gpu-ids 0 --log-dir runs/RL_train --save-model-dir trained_models/RL --data-dir /opt/data/private/datasets/AI2Thor_offline_data_2.0.2/`  
## note
You can see the results of each test in 'eval_results/log.txt', and the iteration process of the model in '/result_epoch.txt'.



