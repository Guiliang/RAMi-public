import numpy as np
import torch
from datetime import datetime

from config.mimic_config import DRLMimicConfig
from data_disentanglement.disentanglement import Disentanglement
from utils.general_utils import return_data


class flappybird_prob:
    """
    An MDP. Contains methods for initialisation, state transition.
    Can be aggregated or unaggregated.
    """

    def __init__(self, gamma=1, image_type=None):
        # assert games_directory is not None
        # self.games_directory = games_directory
        self.gamma = gamma
        self.reset = None
        self.isEpisodic = True

        # same action for all instances
        self.actions = ['the_action']

        self.dimNames = []
        self.dimSizes = []

        if image_type == 'latent':
            data_dim = 48
        else:
            raise ValueError("Unknown image type {0}".format(image_type))

        for i in range(data_dim):
            if image_type == 'latent':
                self.dimNames.append('latent_{}'.format(i))
            self.dimSizes.append('continuous')

        self.stateFeatures = dict(zip(self.dimNames, self.dimSizes))
        self.nStates = len(self.stateFeatures)

        return


def data_builder(episode_number,
                 data_save_dir, dientangler,
                 image_type,
                 game_name, data_loader,
                 action_number,
                 iteration_number=0,
                 disentangler_type='CVAE'):
    memory = []

    # action_number = 2
    # image_type = 'origin'

    def gather_data_values(action_value):
        action_value_items = action_value.split(',')
        action_index = int(action_value_items[0])
        action_values_list = np.zeros([action_number])
        value = 0
        if game_name == 'flappybird':
            for i in range(action_number):
                action_values_list[i] = float(action_value_items[i + 1])
        else:
            raise ValueError('Unknown game {0}'.format(game_name))
        reward = float(action_value_items[-1])
        if reward > 1:
            reward = 1
        return action_index, action_values_list, reward, value

    with open(data_save_dir + '/' + game_name + '/action_values.txt', 'r') as f:
        action_values = f.readlines()

    [action_index_t0, action_values_list_t0,
     reward_t0, value_t0] = gather_data_values(action_values[iteration_number])
    x_t0 = data_loader.dataset.__getitem__(iteration_number)[0]
    conds_t0 = data_loader.dataset.__getitem__(iteration_number)[1][:-1]
    cumu_reward_t0 = conds_t0[-1].item()
    if image_type == "latent":
        x_t0 = x_t0.unsqueeze(0)
        conds_t0 = conds_t0.unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            if disentangler_type == 'Monet':
                dientangler.cmonet.forward(x_t0)
                z0 = dientangler.cmonet.z_mu.flatten()
            elif disentangler_type == 'CMonet':
                dientangler.cmonet.forward(x_t0, conds_t0)
                z0 = dientangler.cmonet.z_mu.flatten()
            else:
                raise ValueError("Unknown disentangler_type {0}".format(disentangler_type))
            z0 = z0.squeeze().cpu().numpy()
    else:
        raise ValueError("Unknown data loader target {0}".format(image_type))

    data_length = 1000 * episode_number - iteration_number
    while len(memory) < data_length and len(memory) < len(action_values) - 1:
        [action_index_t1, action_values_list_t1,
         reward_t1, value_t1] = gather_data_values(action_values[iteration_number + 1])
        if game_name == 'flappybird':
            delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0] + reward_t0
        else:
            raise ValueError('Unknown game {0}'.format(game_name))
        x_t1 = data_loader.dataset.__getitem__(iteration_number + 1)[0]
        conds_t1 = data_loader.dataset.__getitem__(iteration_number + 1)[1][:-1]
        cumu_reward_t1 = conds_t1[-1].item()
        temp = data_loader.dataset.__getitem__(iteration_number)[1][-1]
        assert abs(round(delta, 4) - round(temp.item(), 4)) < 0.001
        if image_type == "latent":
            x_t1 = x_t1.unsqueeze(0)
            conds_t1 = conds_t1.unsqueeze(0).to(torch.float32)
            with torch.no_grad():
                if disentangler_type == 'Monet':
                    dientangler.cmonet.forward(x_t1)
                    z1 = dientangler.cmonet.z_mu.flatten()
                elif disentangler_type == 'CMonet':
                    dientangler.cmonet.forward(x_t1, conds_t1)
                    z1 = dientangler.cmonet.z_mu.flatten()
                else:
                    raise ValueError("Unknown disentangler_type {0}".format(disentangler_type))
            z1 = z1.squeeze().cpu().numpy()
            memory.append([z0, action_index_t0, cumu_reward_t0, z1, delta])
            z0 = z1
        else:
            raise ValueError("Unknown data loader target {0}".format(image_type))
        action_index_t0 = action_index_t1
        action_values_list_t0 = action_values_list_t1
        cumu_reward_t0 = cumu_reward_t1
        iteration_number += 1
        reward_t0 = reward_t1

    print('loading finished')
    return memory


def write_header(writer, image_type):
    problem = flappybird_prob(image_type=image_type)
    headers = []
    headers.append('impact')
    headers = headers + problem.dimNames
    hearder_strings = ', '.join(headers)

    writer.write(hearder_strings + ', action' + ', cumu_reward' + '\n')


def write_data_text(data, writer):
    for i in range(len(data)):
        impact = str(data[i][-1])
        action = str(data[i][1])
        acumu_r = str(data[i][2])
        pixels = data[i][0]
        pixel_string = ', '.join(map(str, pixels))

        writer.write(impact.strip() +
                     ', ' + (pixel_string.strip()) +
                     ', ' + (action.strip()) +
                     ', ' + (acumu_r.strip()) +
                     '\n')


# do not run if called by another file


def run_static_data_generation(save_disentangler_dir=None,
                               game_name="flappybird",
                               disentangler_type='CMonet',
                               image_type='latent',
                               global_model_data_path='',
                               run_tmp_test=False,
                               device='cpu',
                               test_run=5):
    """
    sample latent variables from the disentangled latent distribution
    """

    # data_save_dir = '../example_data/flappybird/origin/latent_features/'
    data_save_dir = '../example_data/'
    if save_disentangler_dir is not None:
        assert disentangler_type in save_disentangler_dir

    mimic_config_path = "../environment_settings/{0}_config.yaml".format(game_name)
    mimic_config = DRLMimicConfig.load(mimic_config_path)

    data_loader = return_data(mimic_config.DEG.Learn,
                              global_model_data_path,
                              mimic_config.DRL.Learn.actions,
                              image_type='origin')
    disentangler = Disentanglement(mimic_config, device, disentangler_type.upper(), True, global_model_data_path)
    if save_disentangler_dir is not None:
        disentangler.load_checkpoint(ckptname=save_disentangler_dir, testing_flag=True, log_file=None)

    training_data_action = data_builder(episode_number=40,
                                        data_save_dir=data_save_dir,
                                        dientangler=disentangler,
                                        image_type=image_type,
                                        game_name=game_name,
                                        iteration_number=0,
                                        disentangler_type=disentangler_type,
                                        data_loader=data_loader,
                                        action_number=mimic_config.DRL.Learn.actions)
    adv_file_name_training = 'adv_training_{4}_data_{1}_{3}_expand.csv'.format(
        image_type, game_name, '', disentangler_type, image_type)
    adv_file_Writer_training = open('../example_data/flappybird/origin/latent_features/' +
                                       adv_file_name_training, 'w')

    print('Writing training csv...')
    write_header(adv_file_Writer_training, image_type=image_type)
    write_data_text(training_data_action, adv_file_Writer_training)
    adv_file_Writer_training.close()

    iteration_number = 1000 * 45
    testing_data_action = data_builder(episode_number=50,
                                       data_save_dir=data_save_dir,
                                       dientangler=disentangler,
                                       image_type=image_type,
                                       game_name=game_name,
                                       iteration_number=iteration_number,
                                       disentangler_type=disentangler_type,
                                       data_loader=data_loader,
                                       action_number=mimic_config.DRL.Learn.actions)

    for i in range(test_run):
        testing_data_action_iter = testing_data_action[i * 1000:(i + 5) * 1000]
        adv_file_name_testing = 'adv_testing_{4}_data_{1}_{3}_run{6}.csv'.format(
            image_type, game_name, '', disentangler_type, image_type, '', i)
        adv_file_Writer_testing = open('../LMUT_data/' + adv_file_name_testing, 'w')

        print('Writing testing csv in iter {0}...'.format(i))
        write_header(adv_file_Writer_testing, image_type=image_type)
        write_data_text(testing_data_action_iter, adv_file_Writer_testing)
        adv_file_Writer_testing.close()


if __name__ == '__main__':
    run_static_data_generation(run_tmp_test=False)
