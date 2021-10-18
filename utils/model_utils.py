import math
from PIL import Image

import torch
import torchvision.utils as tu
import torchvision.transforms.functional as ttf


def get_same_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
    print('padding is {0}'.format(str(padding)))
    return padding


def calculate_conv_output_dimension(size, kernel_size, stride, dilation, padding):
    return math.floor((size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def handle_image_input(img_colored,
                       if_print_img=False,
                       if_binarize=True):
    img_colored = Image.fromarray(img_colored)
    img_colored_resized = ttf.resize(img_colored, size=(84, 84))
    # Image._show(img_colored_resized)
    # img_colored = ttf.rotate(img_colored, angle=270)
    # img_colored = ttf.hflip(img_colored)
    img_gray = ttf.to_grayscale(img_colored_resized, num_output_channels=1)
    # Image._show(img_gray)
    x_t = ttf.to_tensor(img_gray)

    # Apply threshold
    max_value = torch.max(x_t)
    min_value = torch.min(x_t)
    if if_binarize:
        # x_t = x_t > (max_value - min_value) / 2  # mean value
        x_t = x_t > min_value
        x_t = x_t * 255
        x_t = x_t.float()
    if if_print_img:
        import matplotlib.pyplot as plt
        x_t_image = x_t.numpy()
        plt.figure()
        plt.imshow(x_t_image[0])

    return x_t


def store_state_action_data(img_colored, action_values, reward, action_index,
                            save_image_path, action_values_file,
                            game_name, iteration_number):
    action_values_str = str(action_index)+','
    for action_value in action_values:
        action_values_str += str(action_value) + ','
    action_values_str += str(reward) + '\n'
    action_values_file.write(action_values_str)

    img_colored = Image.fromarray(img_colored)
    if game_name == "flappybird":
        img_colored_save = ttf.rotate(img_colored, angle=270)
        img_colored_save = ttf.hflip(img_colored_save)
        img_colored_save = ttf.resize(img_colored_save, size=(128, 128))
        # img_colored_save.show()
    else:
        img_colored_save = img_colored

    origin_save_dir = save_image_path + 'origin/images/{0}-{1}_action{2}_origin.png'.format(game_name,
                                                                                            iteration_number,
                                                                                            action_index)
    tu.save_image(ttf.to_tensor(img_colored_save), open(origin_save_dir, 'wb'))


def visualize_tensor(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    from torchvision.utils import make_grid
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.show()
    return im


def tree_construct_loss(leaf_number):
    entropy_prob = leaf_number / (2 * leaf_number - 1)
    big_o = 1/leaf_number
    structure_cost = math.log((2 * leaf_number - 1) ** 2 / ((leaf_number ** 1.5) * (leaf_number - 1) ** 0.5)) + \
                     (2 * leaf_number - 1) * (-entropy_prob * math.log(entropy_prob) - (1 - entropy_prob) * math.log(1 - entropy_prob))+\
                     big_o
    return structure_cost