import os
import random
from collections import deque
import numpy as np
import torch
import torch.optim as optim

from utils.general_utils import mkdirs
from utils.memory_utils import PrioritizedReplay
from utils.model_utils import handle_image_input, store_state_action_data


class DRLDataGenerator():
    def __init__(self, game_name, config, global_model_data_path, local_test_flag, require_env=True):
        if not local_test_flag:
            mkdirs(global_model_data_path + config.DRL.Learn.data_save_path)
        self.game_name = game_name
        self.data_save_path = global_model_data_path + config.DRL.Learn.data_save_path
        self.config = config
        self.global_iter = 0
        self.ckpt_dir = global_model_data_path + self.config.DRL.Learn.ckpt_dir
        self.ckpt_save_iter = self.config.DRL.Learn.ckpt_save_iter
        if not local_test_flag:
            mkdirs(self.ckpt_dir)

        self.apply_prioritize_memory = False
        if self.apply_prioritize_memory:
            self.memory = PrioritizedReplay(capacity=self.config.DRL.Learn.replay_memory_size)
        else:
            # store the previous observations in replay memory
            self.memory = deque()

        assert game_name == 'flappybird'
        from data_generator.nn_drl.dqn_fb import FlappyBirdDQN
        use_cuda = config.DRL.Learn.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.actions_number = self.config.DRL.Learn.actions
        self.nn = FlappyBirdDQN().to(self.device)
        self.optim = optim.Adam(self.nn.parameters(), lr=self.config.DRL.Learn.learning_rate)
        if config.DRL.Learn.ckpt_load:
            self.load_checkpoint(model_name='flappy_bird_model')
        if require_env:
            from data_generator.fb_game.flappy_bird import FlappyBird
            self.game_state = FlappyBird()

    def load_checkpoint(self, model_name):
        filepath = os.path.join(self.ckpt_dir, model_name)
        if os.path.isfile(filepath):
            if self.device == 'cuda':
                with open(filepath, 'rb') as f:
                    checkpoint = torch.load(f)
            else:
                with open(filepath, 'rb') as f:
                    checkpoint = torch.load(f, map_location=torch.device('cpu'))
            self.global_iter = checkpoint['iter']
            self.nn.load_state_dict(checkpoint['model_states']['FlappyBirdDQN'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim_DQN'])
            print("load from checkpoint: {0}".format(filepath))

    def sample_batch(self):
        if self.apply_prioritize_memory:
            minibatch, idxs, is_weights = self.memory.sample(self.config.DRL.Learn.batch)
        else:
            minibatch = random.sample(self.memory, self.config.DRL.Learn.batch)
            is_weights = np.ones(shape=[self.config.DRL.Learn.batch])
            idxs = None

        return minibatch, idxs, is_weights

    def test_model_and_generate_data(self, test_size=500):
        """
        Generate the data with the learned model, here we use flappybird to show an example,
        You replace it with other agent, for example, the agent in tensorpack http://models.tensorpack.com/#OpenAIGym
        """
        assert self.game_name == "flappybird"
        with open(self.data_save_path + 'action_values.txt', 'w') as action_values_file:
            action_index = 0
            x_t0_colored, r_t, terminal = self.game_state.next_frame(action_index)
            x_t0 = handle_image_input(x_t0_colored[:self.game_state.screen_width, :int(self.game_state.base_y)])
            s_t0 = torch.cat(tuple(x_t0 for _ in range(4))).to(self.device)
            # self.global_iter += 1
            while self.global_iter < test_size:
                with torch.no_grad():
                    readout = self.nn(s_t0.unsqueeze(0))
                readout = readout.cpu().numpy()
                action_index = np.argmax(readout)
                x_t1_colored, r_t, terminal = self.game_state.next_frame(action_index)
                store_state_action_data(
                    img_colored=x_t0_colored[:self.game_state.screen_width, :int(self.game_state.base_y)],
                    action_values=readout[0], reward=r_t, action_index=action_index,
                    save_image_path=self.data_save_path, action_values_file=action_values_file,
                    game_name=self.game_name, iteration_number=self.global_iter)
                print("finishing save data iter {0}".format(self.global_iter))
                x_t1 = handle_image_input(x_t1_colored[:self.game_state.screen_width, :int(self.game_state.base_y)])
                s_t1 = torch.cat((s_t0[1:, :, :], x_t1.to(self.device)))
                s_t0 = s_t1
                x_t0_colored = x_t1_colored
                self.global_iter += 1

