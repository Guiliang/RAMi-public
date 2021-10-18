import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def gather_data_values(action_value, action_number, game_name):
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


def calculate_cumulative_reward(reward_all, home_away_identifier=None):
    """
    calculate the condition R (cumulative reward)
    :param home_away_identifier: 0 away team / 1 home team
    :param reward_all: 0 no score /1 score
    :return: the cumulative reward
    """
    data_length = len(reward_all)
    cumulative_reward = 0
    cumulative_reward_all = []
    for i in range(data_length):
        if reward_all[i]:
            if home_away_identifier is not None:
                if home_away_identifier[i]:
                    cumulative_reward += 1
                else:
                    cumulative_reward -= 1
            else:
                cumulative_reward += reward_all[i]
        cumulative_reward_all.append(cumulative_reward)

    return np.asarray(cumulative_reward_all)


class CustomImageConditionFolder(ImageFolder):
    """
    Read and return the training data.
    """

    def __init__(self, root, cond_dir, img_path, image_type, game, action_number, transform=None):
        super(CustomImageConditionFolder, self).__init__(root, transform)
        self.indices = range(len(self) - 1)
        self.game_name = game
        self.action_number = action_number
        self.img_path = img_path
        self.img_type = image_type
        with open(cond_dir, 'r') as f:
            self.action_values = f.readlines()

        rewards = []
        for action_value in self.action_values:
            reward = float(action_value.split(',')[-1])
            rewards.append(reward)

        self.cumu_reward_all = calculate_cumulative_reward(reward_all=rewards)

    def __getitem__(self, index1):
        if index1 == len(self) - 1:  # handle the last index
            index1 -= 1
        index2 = random.choice(self.indices)
        try:
            [action_index_t0_i1, action_values_list_t0_i1,
             reward_t0_i1, value_t0_i1] = gather_data_values(self.action_values[index1], self.action_number, self.game_name)
        except:
            print(index1)
        [action_index_t1_i1, action_values_list_t1_i1,
         reward_t1_i1, value_t1_i1] = gather_data_values(self.action_values[index1 + 1], self.action_number,
                                                         self.game_name)
        [action_index_t0_i2, action_values_list_t0_i2,
         reward_t0_i2, value_t0_i2] = gather_data_values(self.action_values[index2], self.action_number, self.game_name)
        [action_index_t1_i2, action_values_list_t1_i2,
         reward_t1_i2, value_t1_i2] = gather_data_values(self.action_values[index2 + 1], self.action_number,
                                                         self.game_name)

        path1 = self.img_path + '/images/{0}-{1}_action{2}_{3}.png'.format(self.game_name, index1, action_index_t0_i1,
                                                                           self.img_type)
        path2 = self.img_path + '/images/{0}-{1}_action{2}_{3}.png'.format(self.game_name, index2, action_index_t0_i2,
                                                                           self.img_type)
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.game_name == 'flappybird':
            delta_i1 = max(action_values_list_t1_i1) - action_values_list_t0_i1[action_index_t0_i1] + reward_t0_i1
            delta_i2 = max(action_values_list_t1_i2) - action_values_list_t0_i2[action_index_t0_i2] + reward_t0_i2
        else:
            raise ValueError('Unknown game {0}'.format(self.game_name))
        action_t0_i1 = [0 for i in range(self.action_number)]
        action_t0_i1[action_index_t0_i1] = 1
        action_t0_i2 = [0 for i in range(self.action_number)]
        action_t0_i2[action_index_t0_i2] = 1
        cond_i1 = torch.tensor(action_t0_i1 + [reward_t0_i1, delta_i1])
        cond_i2 = torch.tensor(action_t0_i2 + [reward_t0_i2, delta_i2])

        return img1, cond_i1, img2, cond_i2


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def return_data(config, global_model_data_path, action_number, image_type=None):
    if image_type is None:
        image_type = config.image_type

    name = config.name
    dset_dir = global_model_data_path + config.dset_dir
    batch_size = config.batch_size
    transform = transforms.Compose([
        # transforms.Resize((image_length, image_width)),
        transforms.ToTensor(), ])

    if name.lower() == 'flappybird':
        root = os.path.join(dset_dir, 'flappybird/' + image_type)
        cond_dir = os.path.join(dset_dir, 'flappybird/action_values.txt')
        img_path = os.path.join(dset_dir, 'flappybird/' + image_type)
        train_kwargs = {'root': root, 'transform': transform, 'cond_dir': cond_dir,
                        'img_path': img_path, 'image_type': image_type,
                        'game': 'flappybird', 'action_number': action_number}
        dset = CustomImageConditionFolder
    else:
        raise NotImplementedError

    print("reading img from {0}".format(img_path), flush=True)
    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True
                              )

    data_loader = train_loader
    return data_loader


def handle_dict_list(dict_list_A, dict_list_B, option):
    for key in dict_list_B.keys():
        list_B = dict_list_B.get(key)
        if key in dict_list_A.keys():
            list_A = dict_list_A.get(key)
            if option == 'add':
                list_new = list(set(list_A + list_B))
            elif option == 'substract':
                list_new = list(set(list_A) - set(list_B))
            else:
                raise ValueError('unknown option {0}'.format(option))
            dict_list_A[key] = list_new
        else:
            if option == 'add':
                dict_list_A[key] = list_B
    return dict_list_A


