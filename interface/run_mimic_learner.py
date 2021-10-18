import os
import optparse
import traceback

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from config.mimic_config import DRLMimicConfig
from mimic_learner.learner import MimicLearner

optparser = optparse.OptionParser()
optparser.add_option("-r", "--round number", dest="ROUND_NUMBER", default=None,
                     help="round number of mcts (default = None)")
optparser.add_option("-d", "--log dir", dest="LOG_DIR", default=None,
                     help="the dir of log")
optparser.add_option("-g", "--game name", dest="GAME_NAME", default='flappybird',
                     help="the name of running game")
optparser.add_option("-m", "--method name", dest="METHOD_NAME", default='mcts',
                     help="the name of applied method")
optparser.add_option("-l", "--launch time", dest="LAUNCH_TIME", default=None,
                     help="the time we launch this program")
optparser.add_option("-c", "--cpuct", dest="C_PUCT", default=None,
                     help="cpuct")
optparser.add_option("-p", "--play", dest="PLAY", default=None,
                     help="play number")
optparser.add_option("-n", "--dientangler_name", dest="DEG", default="CMONET",
                     help="play number")
# optparser.add_option("-d", "--dir of just saved mcts", dest="MCTS_DIR", default=None,
#                      help="dir of just saved mcts (default = None)")
opts = optparser.parse_args()[0]


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if opts.PLAY is not None:
        play = int(opts.PLAY)
    else:
        play = None

    if opts.C_PUCT is not None:
        c_puct = float(opts.C_PUCT)
    else:
        c_puct = None
    mimic_config_path = "../environment_settings/{0}_config.yaml".format(opts.GAME_NAME)
    mimic_config = DRLMimicConfig.load(mimic_config_path)
    global_model_data_path = '../'

    print('global path is : {0}'.format(global_model_data_path))
    if opts.LOG_DIR is not None:
        if os.path.exists(opts.LOG_DIR):
            log_file = open(opts.LOG_DIR, 'a')
        else:
            log_file = open(opts.LOG_DIR, 'w')
    else:
        log_file = None

    try:
        print("\nRunning for game {0} with {1}".format(opts.GAME_NAME, opts.METHOD_NAME), file=log_file)
        mimic_learner = MimicLearner(game_name=opts.GAME_NAME,
                                     method=opts.METHOD_NAME,
                                     disentangler_name=opts.DEG,
                                     config=mimic_config,
                                     global_model_data_path=global_model_data_path,
                                     )
        shell_round_number = int(opts.ROUND_NUMBER) if opts.ROUND_NUMBER is not None else None

        mimic_learner.train_mimic_model(
            shell_round_number=shell_round_number,
            log_file=log_file,
            launch_time=opts.LAUNCH_TIME,
            disentangler_name=opts.DEG,
            data_type='latent',
            c_puct=c_puct,
            play=play, )

        if log_file is not None:
            log_file.close()
    except Exception as e:
        traceback.print_exc(file=log_file)
        if log_file is not None:
            log_file.write(str(e))
            log_file.flush()
            log_file.close()
        # sys.stderr.write('finish shell round {0}'.format(shell_round_number))


if __name__ == "__main__":
    run()
    exit(0)
