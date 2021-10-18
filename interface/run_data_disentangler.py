import os

import torch

cwd = os.getcwd()
import sys
sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from config.mimic_config import DRLMimicConfig
from data_disentanglement.disentanglement import Disentanglement


def run():
    game_name = 'flappybird'
    deg_type = 'CMONET'
    if game_name == 'flappybird':
        config_path = "../environment_settings/flappybird_config.yaml"
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    print("Running environment {0}".format(game_name))
    deg_config = DRLMimicConfig.load(config_path)
    use_cuda = torch.cuda.is_available()
    print("find gpu :{0}".format(use_cuda), flush=True)
    if not use_cuda:
        device = 'cpu'
    else:
        device = 'cuda'
    local_test_flag = True
    global_model_data_path = ''
    DEG = Disentanglement(config=deg_config, device=device, deg_type=deg_type,
                          global_model_data_path=global_model_data_path, local_test_flag=local_test_flag)
    # if local_test_flag:
    #     DEG.ckpt_dir = '../saved_models/DEG/{0}/'.format(game_name)
    running_label = ''
    print("launching on machine: {0} with max iter: {1} and running label: {2}".format(global_model_data_path,
                                                                                       DEG.max_iter,
                                                                                       running_label), flush=True)

    if deg_type == 'MONET':
        DEG.train_monet(running_label=running_label)
    elif deg_type == 'CMONET':
        DEG.train_cmonet(running_label=running_label)
    else:
        raise ValueError('Unknown deg type {0}'.format(deg_type))


if __name__ == "__main__":
    run()
