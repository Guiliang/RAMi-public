"""solver.py"""

import os
import torch
from torchvision.utils import save_image

from data_disentanglement.nn_deg.conditional_monet import CMONetModel
from data_disentanglement.nn_deg.monet import MONetModel
from utils.general_utils import mkdirs, return_data


class Disentanglement(object):
    def __init__(self, config, device, deg_type,
                 local_test_flag=False,
                 global_model_data_path='',
                 apply_data_loader=True):

        self.global_model_data_path = global_model_data_path

        self.device = device
        print("Using {0}".format(self.device))
        self.name = config.DEG.Learn.name
        self.max_iter = config.DEG.Learn.max_iter
        self.print_iter = config.DEG.Learn.print_iter
        self.global_iter = 0
        self.pbar = None

        # Data
        self.batch_size = config.DEG.Learn.batch_size
        if apply_data_loader:
            self.data_loader = return_data(config.DEG.Learn, global_model_data_path, config.DRL.Learn.actions)
        self.action_size = config.DRL.Learn.actions

        # Dimension
        self.z_dim = config.DEG.Learn.z_dim
        self.image_length = config.DEG.Learn.image_length
        self.image_width = config.DEG.Learn.image_width

        self.nc = 3
        if deg_type == 'MONET':
            self.z_dim = config.DEG.MONET.z_dim * config.DEG.MONET.num_slots
            self.monet = MONetModel(num_slots=config.DEG.MONET.num_slots,
                                    z_dim=config.DEG.MONET.z_dim,
                                    lr=config.DEG.MONET.lr,
                                    beta=config.DEG.MONET.beta,
                                    gamma=config.DEG.MONET.gamma,
                                    device=self.device)
            self.nets = [self.monet.netAttn, self.monet.netCVAE]

            # Checkpoint
            self.output_dir = os.path.join(self.global_model_data_path + config.DEG.MONET.output_dir, 'output')
            self.ckpt_dir = os.path.join(self.global_model_data_path + config.DEG.MONET.ckpt_dir)
            self.ckpt_save_iter = config.DEG.MONET.ckpt_save_iter
            if not local_test_flag and not os.path.exists(self.ckpt_dir):
                mkdirs(self.ckpt_dir)
            self.output_save = config.DEG.MONET.output_save
        elif deg_type == 'CMONET':
            self.z_dim = config.DEG.MONET.z_dim * config.DEG.CMONET.num_slots
            self.cmonet = CMONetModel(num_slots=config.DEG.CMONET.num_slots,
                                      z_dim=config.DEG.CMONET.z_dim,
                                      lr=config.DEG.CMONET.lr,
                                      condition_size=config.DEG.CMONET.condition_size,
                                      beta=config.DEG.CMONET.beta,
                                      gamma=config.DEG.CMONET.gamma,
                                      device=self.device)
            self.nets = [self.cmonet.netAttn, self.cmonet.netCVAE]

            # Checkpoint
            self.output_dir = os.path.join(self.global_model_data_path + config.DEG.CMONET.output_dir, 'output')
            self.ckpt_dir = os.path.join(self.global_model_data_path + config.DEG.CMONET.ckpt_dir)
            self.ckpt_save_iter = config.DEG.CMONET.ckpt_save_iter
            if not local_test_flag and not os.path.exists(self.ckpt_dir):
                mkdirs(self.ckpt_dir)
            self.output_save = config.DEG.CMONET.output_save
        else:
            raise ValueError('Unknown deg type {0}'.format(deg_type))

    def train_cmonet(self, running_label=None):
        self.net_mode(train=True)
        out = False
        while not out:
            for x_true1, cond_y_1, x_true2, cond_y_2 in self.data_loader:
                x_true1 = torch.squeeze(x_true1, 0).float()
                cond1 = torch.squeeze(cond_y_1[:, :self.action_size + 1], 0).float()

                if self.device == 'cuda':
                    x_true1 = x_true1.to(self.device)
                    cond1 = cond1.to(self.device)
                self.cmonet.optimize_parameters(x_true1, cond1)

                if self.global_iter % self.print_iter == 0:

                    print("CMONET: Loss Decoder {0}, Loss Encoder {1}, "
                          "Loss Mask {2}, Pixel level loss {3},"
                          "generated images max: {4}/ min:{5}.".format(self.cmonet.loss_D,
                                                                       self.cmonet.loss_E,
                                                                       self.cmonet.loss_mask,
                                                                       -torch.logsumexp(self.cmonet.b, dim=1).mean(),
                                                                       torch.max(self.cmonet.x0),
                                                                       torch.min(self.cmonet.x0)
                                                                       ),
                          flush=True)
                    visual_ret = {}
                    save_image(x_true1[0], open("./cmonet_img/{0}/tmp_cmonet_ture_iter{1}.png".format(
                        self.name, self.global_iter), "wb"))
                    for visual_name in self.cmonet.visual_names:
                        if isinstance(visual_name, str):
                            visual_ret.update({visual_name: getattr(self.cmonet, visual_name)})
                            save_image(visual_ret[visual_name][0],
                                       open("./cmonet_img/{5}/cmonet_{0}_s{1}b{2}g{3}_iter{4}.png".format(
                                           visual_name,
                                           self.cmonet.num_slots,
                                           self.cmonet.beta,
                                           self.cmonet.gamma,
                                           self.global_iter,
                                           self.name),
                                           "wb"))

                if self.global_iter % self.ckpt_save_iter == 0:
                    print('Saving MONET models', flush=True)
                    if running_label is None:
                        running_tag = ''
                    else:
                        running_tag = running_label
                    self.save_checkpoint('{4}CMonet-slot{0}-beta{1}-gamma{2}-{3}.pt'.format(self.cmonet.num_slots,
                                                                                            self.cmonet.beta,
                                                                                            self.cmonet.gamma,
                                                                                            self.global_iter,
                                                                                            running_tag),
                                         type='CMonet', verbose=True)

                if self.global_iter >= self.max_iter:
                    out = True
                    break
                self.global_iter += 1

    def train_monet(self, running_label=None):
        self.net_mode(train=True)
        out = False
        while not out:
            for x_true1, cond_y_1, x_true2, cond_y_2 in self.data_loader:
                x_true1 = torch.squeeze(x_true1, 0).float()
                if self.device == 'cuda':
                    x_true1 = x_true1.to(self.device)
                self.monet.optimize_parameters(x_true1)

                if self.global_iter % self.print_iter == 0:
                    print("MONET: Loss Decoder {0}, Loss Encoder {1}, "
                          "Loss Mask {2}, Pixel level loss {3},"
                          "generated images max: {4}/ min:{5}.".format(self.monet.loss_D,
                                                                       self.monet.loss_E,
                                                                       self.monet.loss_mask,
                                                                       -torch.logsumexp(self.monet.b, dim=1).mean(),
                                                                       torch.max(self.monet.x0),
                                                                       torch.min(self.monet.x0)
                                                                       ),
                          flush=True)
                    visual_ret = {}
                    save_image(x_true1[0],
                               open("./monet_img/{0}/{2}monet_ture_iter{1}.png".format(self.name,
                                                                                       self.global_iter,
                                                                                       running_label), "wb"))
                    for visual_name in self.monet.visual_names:
                        if isinstance(visual_name, str):
                            visual_ret.update({visual_name: getattr(self.monet, visual_name)})
                            save_image(visual_ret[visual_name][0],
                                       open("./monet_img/{5}/"
                                            "{6}monet_{0}_s{1}b{2}g{3}_iter{4}.png".format(visual_name,
                                                                                           self.monet.num_slots,
                                                                                           self.monet.beta,
                                                                                           self.monet.gamma,
                                                                                           self.global_iter,
                                                                                           self.name,
                                                                                           running_label),
                                            "wb"))
                if self.global_iter % self.ckpt_save_iter == 0:
                    print('Saving MONET models', flush=True)
                    self.save_checkpoint('{4}Monet-slot{0}-beta{1}-gamma{2}-{3}.pt'.format(self.monet.num_slots,
                                                                                           self.monet.beta,
                                                                                           self.monet.gamma,
                                                                                           self.global_iter,
                                                                                           running_label),
                                         type='Monet', verbose=True)

                if self.global_iter >= self.max_iter:
                    out = True
                    break
                self.global_iter += 1

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', type=None, verbose=True):

        if type == "Monet":
            model_states = {'netAttn': self.monet.netAttn.state_dict(),
                            'netCVAE': self.monet.netCVAE.state_dict()}
            optim_states = {'optim': self.monet.optimizer.state_dict()}
            states = {'iter': self.global_iter,
                      'model_states': model_states,
                      'optim_states': optim_states}
        elif type == "CMonet":
            model_states = {'netAttn': self.cmonet.netAttn.state_dict(),
                            'netCVAE': self.cmonet.netCVAE.state_dict()}
            optim_states = {'optim': self.cmonet.optimizer.state_dict()}
            states = {'iter': self.global_iter,
                      'model_states': model_states,
                      'optim_states': optim_states}
        else:
            raise EnvironmentError("Saving type {0} is undefined".format(type))

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            # self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))
            print("saved checkpoint '{}' (iter {})".format(filepath, self.global_iter), flush=True)

    def load_checkpoint(self, ckptname, verbose=True, testing_flag=False, log_file=None):

        if not testing_flag:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.max_iter)
        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                if torch.cuda.is_available():
                    checkpoint = torch.load(f)
                else:
                    checkpoint = torch.load(f, map_location=torch.device('cpu'))

            self.global_iter = checkpoint['iter']
            if "CMonet" in ckptname:
                self.cmonet.netCVAE.load_state_dict(checkpoint['model_states']['netCVAE'])
                self.cmonet.netAttn.load_state_dict(checkpoint['model_states']['netAttn'])
                self.cmonet.optimizer.load_state_dict(checkpoint['optim_states']['optim'])
            elif "Monet" in ckptname:
                self.monet.netCVAE.load_state_dict(checkpoint['model_states']['netCVAE'])
                self.monet.netAttn.load_state_dict(checkpoint['model_states']['netAttn'])
                self.monet.optimizer.load_state_dict(checkpoint['optim_states']['optim'])
            if not testing_flag:
                self.pbar.update(self.global_iter)
            if verbose:
                print("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter), file=log_file)
        else:
            if verbose:
                print("=> no checkpoint found at '{}'".format(filepath), file=log_file)
