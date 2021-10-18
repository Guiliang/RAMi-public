import csv
from datetime import datetime
import ast
import os

import logging
import torch
import numpy as np
import torch.nn.functional as F
from mimic_learner.mcts_learner import mcts
# print (mcts.c_PUCT)
from mimic_learner.mcts_learner.mcts import test_mcts, execute_episode_single
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from PIL import Image
import torchvision.transforms.functional as ttf
from utils.general_utils import gather_data_values

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)


class MimicLearner():
    def __init__(self, game_name, method, config,
                 global_model_data_path, disentangler_name, options=[]):
        self.disentangler_name = disentangler_name
        if self.disentangler_name == 'MONET':
            self.z_dim = config.DEG.Learn.z_dim * config.DEG.MONET.num_slots
        elif self.disentangler_name == 'CMONET':
            self.z_dim = config.DEG.Learn.z_dim * config.DEG.CMONET.num_slots
        self.mimic_env = MimicEnv(n_action_types=self.z_dim)
        self.game_name = game_name
        self.action_number = config.DRL.Learn.actions
        self.action_type = config.DRL.Learn.action_type
        self.global_model_data_path = global_model_data_path

        self.num_simulations = config.Mimic.Learn.num_simulations
        self.episodic_sample_number = config.Mimic.Learn.episodic_sample_number
        self.data_save_dir = self.global_model_data_path + config.DEG.Learn.dset_dir
        # self.image_type = config.DEG.Learn.image_type
        self.image_type = None
        self.iteration_number = 0
        self.method = method

        self.memory = None
        self.mcts_saved_dir = config.Mimic.Learn.mcts_saved_dir
        self.shell_saved_model_dir = self.global_model_data_path + self.mcts_saved_dir
        self.max_k = config.Mimic.Learn.max_k

    def static_data_loader(self, log_file, training_flag=True):
        print("Reading from static data", file=log_file)
        self.memory = []
        cwd = os.getcwd()

        if training_flag:
            read_data_dir = cwd.replace('/interface', '') + \
                            '/example_data/flappybird/origin/latent_features/' \
                            'adv_training_latent_data_flappybird_{0}_expand.csv'.format(self.disentangler_name)

        else:
            read_data_dir = cwd.replace('/interface', '') + \
                            '/example_data/flappybird/origin/latent_features/' \
                            'adv_testing_latent_data_flappybird_{0}_expand.csv'.format(self.disentangler_name)

        print("reading data from {0}".format(read_data_dir), file=log_file)
        skip_line = True
        with open(read_data_dir, 'r') as csvfile:
            csv_read_line = csv.reader(csvfile)
            for row in csv_read_line:
                if skip_line:
                    skip_line = False
                    continue
                impact = float(row[0])
                action_id = int(row[-2])
                cumu_reward = float(row[-1])
                z0 = np.asarray([float(i) for i in row[1:-2]])
                self.memory.append([z0, action_id, cumu_reward, impact])

    def generate_prediction_results(self, init_state, init_var_list, moved_nodes, max_node, log_file,
                                    visualize_flag, feature_importance_flag):
        parent_state = init_state
        state = init_state
        total_data_length = len(state[0])
        parent_var_list = init_var_list
        root_binary = BinaryTreeNode(state=init_state[0], level=0, prediction=moved_nodes[0].state_prediction[0])
        binary_node_index = [root_binary]

        if feature_importance_flag:
            feature_importance_all = {}

        for moved_node in moved_nodes[:max_node]:
            selected_action = moved_node.action
            if selected_action is not None:
                state, new_var_list = self.mimic_env.next_state(state=parent_state, action=selected_action,
                                                                parent_var_list=parent_var_list)
                split_index = int(selected_action.split('_')[0])
                split_binary_node = binary_node_index[split_index]
                parent_state = state
                parent_var_list = new_var_list
                split_binary_node.action = selected_action
                split_binary_node.left_child = BinaryTreeNode(state=state[split_index],
                                                              level=split_binary_node.level + 1,
                                                              prediction=moved_node.state_prediction[split_index])
                split_binary_node.right_child = BinaryTreeNode(state=state[split_index + 1],
                                                               level=split_binary_node.level + 1,
                                                               prediction=moved_node.state_prediction[split_index + 1])
                binary_node_index.pop(split_index)
                binary_node_index.insert(split_index, split_binary_node.left_child)
                binary_node_index.insert(split_index + 1, split_binary_node.right_child)

                if feature_importance_flag:
                    feature_dim = selected_action.split('_')[1]
                    parent_impacts = []
                    for index in split_binary_node.state:
                        parent_impacts.append(self.mimic_env.data_all[index][-1])
                    parent_var = np.var(parent_impacts)

                    left_child_impacts = []
                    for index in state[split_index]:
                        left_child_impacts.append(self.mimic_env.data_all[index][-1])
                    left_child_var = np.var(left_child_impacts)

                    right_child_impacts = []
                    for index in state[split_index + 1]:
                        right_child_impacts.append(self.mimic_env.data_all[index][-1])
                    right_child_var = np.var(right_child_impacts)

                    var_reduction = float(len(split_binary_node.state)) / total_data_length * parent_var - \
                                    float(len(state[split_index])) / total_data_length * left_child_var - \
                                    float(len(state[split_index + 1])) / total_data_length * right_child_var

                    if feature_importance_all.get(feature_dim) is not None:
                        feature_importance_all[feature_dim] += var_reduction
                    else:
                        feature_importance_all.update({feature_dim: var_reduction})

        binary_states = []
        binary_predictions = [None for i in range(len(self.mimic_env.data_all))]

        if visualize_flag:
            self.iterative_read_binary_tree(root_binary, log_file, visualize_flag=visualize_flag)
        if feature_importance_flag:
            print(feature_importance_all)

        selected_binary_node_index = binary_node_index

        # state_predictions = []
        # for binary_node in selected_binary_node_index:
        #     state_target_values = []
        #     for data_index in binary_node.state:
        #         state_target_values.append(data[data_index][-1])
        #     state_predictions.append(sum(state_target_values) / len(state_target_values))

        for binary_node in selected_binary_node_index:
            binary_states.append(binary_node.state)
            for data_index in binary_node.state:
                binary_predictions[data_index] = binary_node.prediction

        return_value_log = self.mimic_env.get_return(state=binary_states)
        return_value_log_struct = self.mimic_env.get_return(state=binary_states, apply_structure_cost=True)
        return_value_var_reduction = self.mimic_env.get_return(state=binary_states, apply_variance_reduction=True)

        # predictions = [None for i in range(len(data))]
        # assert len(state) == len(moved_nodes[-1].state_prediction)
        # for subset_index in range(len(state)):
        #     subset = state[subset_index]
        #     for data_index in subset:
        #         predictions[data_index] = moved_nodes[-1].state_prediction[subset_index]
        #
        # for predict_index in range(len(predictions)):
        #     pred_diff = binary_predictions[predict_index] - predictions[predict_index]
        #     print(pred_diff)

        ae_all = []
        se_all = []
        for data_index in range(len(binary_predictions)):
            if binary_predictions[data_index] is not None:
                real_value = self.mimic_env.data_all[data_index][-1]
                predicted_value = binary_predictions[data_index]
                ae = abs(real_value - predicted_value)
                ae_all.append(ae)
                mse = ae ** 2
                se_all.append(mse)
        mae = np.mean(ae_all)
        mse = np.mean(se_all)
        rmse = (mse) ** 0.5
        leaves_number = len(state)
        return return_value_log, return_value_log_struct, return_value_var_reduction, mae, rmse, leaves_number

    def train_mimic_model(self, shell_round_number,
                          log_file, launch_time, data_type,
                          disentangler_name,
                          saved_nodes_dir=None,
                          c_puct=None, play=None):
        if self.method == 'mcts':
            self.static_data_loader(log_file=log_file, training_flag=True)
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state()
            if c_puct is not None:
                mcts.c_PUCT = c_puct
            mcts_saved_dir = self.global_model_data_path + self.mcts_saved_dir
            shell_saved_model_dir = mcts_saved_dir + '_tmp_shell_saved_CPUCT{1}' \
                                                     '_play{2}_{3}.pkl'.format('', c_puct, play, launch_time)
            execute_episode_single(num_simulations=self.num_simulations,
                                   TreeEnv=self.mimic_env,
                                   tree_writer=None,
                                   mcts_saved_dir=mcts_saved_dir,
                                   max_k=self.max_k,
                                   init_state=init_state,
                                   init_var_list=init_var_list,
                                   ignored_dim=[],
                                   shell_round_number=shell_round_number,
                                   shell_saved_model_dir=shell_saved_model_dir,
                                   log_file=log_file,
                                   disentangler_name=disentangler_name,
                                   apply_split_parallel=True,
                                   play=play)

        else:
            raise ValueError('Unknown method {0}'.format(self.method))


class BinaryTreeNode():
    def __init__(self, state, level, prediction):
        self.state = state
        self.left_child = None
        self.right_child = None
        self.action = None
        self.level = level
        self.prediction = prediction
