import yaml
from utils.config_utils import InitWithDict


class DRLMimicConfig(object):
    def __init__(self, init):
        self.DEG = DRLMimicConfig.DEG(init["DEG"])
        self.DRL = DRLMimicConfig.DRL(init["DRL"])
        self.Mimic = DRLMimicConfig.Mimic(init["Mimic"])

    class DEG(InitWithDict):

        def __init__(self, init):
            super(DRLMimicConfig.DEG, self).__init__(init)
            self.MONET = DRLMimicConfig.DEG.MONET(init["MONET"])
            self.CMONET = DRLMimicConfig.DEG.CMONET(init["CMONET"])
            self.Learn = DRLMimicConfig.DEG.Learn(init["Learn"])

        class Learn(InitWithDict):
            batch_size = None
            name = None
            cuda = None
            max_iter = None
            print_iter = None
            z_dim = None
            image_length = None
            image_width = None
            image_type = None
            num_workers = None
            dset_dir = None

        class MONET(InitWithDict):
            lr = None
            num_slots = None
            z_dim = None
            beta = None
            gamma = None

        class CMONET(InitWithDict):
            lr = None
            num_slots = None
            z_dim = None
            beta = None
            gamma = None
            condition_size = None

    class DRL(InitWithDict):

        def __init__(self, init):
            super(DRLMimicConfig.DRL, self).__init__(init)
            self.Learn = DRLMimicConfig.DRL.Learn(init["Learn"])

        class Learn(InitWithDict):
            game = None
            actions = None
            gamma = None
            observe = None
            explore = None
            final_epsilon = None
            initial_epsilon = None
            replay_memory_size = None
            batch = None
            frame_per_action = None
            cuda = None
            ckpt_dir = None
            ckpt_save_iter = None
            max_iter = None
            ckpt_load = None
            beta1_D = None
            beta2_D = None
            viz_ta_iter = None
            action_type = None

    class Mimic(InitWithDict):

        def __init__(self, init):
            super(DRLMimicConfig.Mimic, self).__init__(init)
            self.Learn = DRLMimicConfig.Mimic.Learn(init["Learn"])

        class Learn(InitWithDict):
            num_simulations = None
            replay_memory_size = None
            episodic_sample_number = None
            max_k = None
            mcts_saved_dir = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return DRLMimicConfig(config)
