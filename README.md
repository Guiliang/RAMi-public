# RAMi-public
```
conda create -n rami-venv python=3.7
conda activate rami-venv
conda install --file requirements.txt
conda install pytorch torchvision torchaudio -c pytorch
```
In this code, we use the flappybird agent as an example. You can replace it with other agent, for example, the agent in tensorpack http://models.tensorpack.com/#OpenAIGym

## Data Generation
```
# creat data folder
mkdir ./example_data/
mkdir ./example_data/flappybird/
mkdir ./example_data/flappybird/origin/
mkdir ./example_data/flappybird/origin/images/
mkdir ./example_data/flappybird/origin/latent_features/
# run model
python ./interface/run_data_generator.py
```

## Representation Model
```
# creat data and model folder
mkdir ./interface/cmonet_img
mkdir ./interface/cmonet_img/flappybird/
mkdir ./interface/monet_img
mkdir ./interface/monet_img/flappybird/
mkdir ./saved_models/
mkdir ./saved_models/DEG/
mkdir ./saved_models/DEG/flappybird
# run model
python ./interface/run_data_disentangler.py
```

## mimic model
```
# creat data and model folder
mkdir ./saved_models/DRL-interpreter-model/
mkdir ./saved_models/DRL-interpreter-model/MCTS/
mkdir ./saved_models/DRL-interpreter-model/MCTS/flappybird/
mkdir ./saved_models/DRL-interpreter-model/MCTS/flappybird/tree_models/
mkdir ./saved_models/DRL-interpreter-model/MCTS/flappybird/tree_nodes/
# run model
./run_mcts_mimic.sh
```


