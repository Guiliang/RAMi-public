import os
cwd = os.getcwd()
import sys
sys.path.append(cwd.replace('/interface', ''))
print (sys.path)
from config.mimic_config import DRLMimicConfig
from data_generator.generator import DRLDataGenerator


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    game_name = 'flappybird'
    print('Running game {0}'.format(game_name))
    if game_name == 'flappybird':
        mimic_env_config_path = "../environment_settings/" \
                                 "flappybird_config.yaml"
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    mimic_config = DRLMimicConfig.load(mimic_env_config_path)

    local_test_flag = True
    if local_test_flag:
        mimic_config.DRL.Learn.data_save_path = '../example_data/flappybird/'
        mimic_config.DRL.Learn.ckpt_dir = '../data_generator/saved_models/'
        global_model_data_path = ''
    else:
        raise EnvironmentError("Unknown running setting, please set up your own environment")


    data_generator = DRLDataGenerator(game_name=game_name, config=mimic_config,
                                      global_model_data_path=global_model_data_path, local_test_flag=local_test_flag)
    data_generator.test_model_and_generate_data()


if __name__ == "__main__":
    run()
