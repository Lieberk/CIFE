import numpy as np
import random
import os
import time
import math
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def data_clean(line):
    line = line.lower().strip()
    line = line.replace('&nbsp;', '。')
    line = line.replace(' ', '')
    line = line.replace('\t', '')

    if len(line) == 0:
        line = '空'

    return line


def matrix_mask(mask_2d):
    l = mask_2d.size(1)
    mask_3d_expanded = mask_2d.unsqueeze(1).repeat(1, l, 1)
    swapped_matrix = mask_3d_expanded.permute(0, 2, 1)
    rotated_matrix = swapped_matrix.flip([2])
    return mask_3d_expanded * rotated_matrix


def make_exp_dirs(exp_name):
    day_logs_root = 'generation_logs/' + time.strftime("%Y-%m%d", time.localtime())
    os.makedirs(day_logs_root, exist_ok=True)
    exp_log_path = os.path.join(day_logs_root, exp_name)

    # model_save_root ='saved_models/'
    # model_save_path = os.path.join(model_save_root, exp_name)

    os.makedirs(exp_log_path, exist_ok=True)  # log dir make
    # os.makedirs(model_save_path, exist_ok=True)  # model save dir make

    return exp_log_path


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def writr_gt(test_dataloader, log_dir, tkr):
    gt_file_name_test = os.path.join(log_dir, 'gt4test.txt')
    gt_txt_test = open(gt_file_name_test, 'w', encoding="utf-8")

    for idx, test_data in tqdm(enumerate(test_dataloader)):
        for i in range(len(test_data['target_ids'])):
            # print(test_data['input_ids'])
            label_pad = test_data['target_ids'][i].masked_fill(test_data['target_ids'][i] == -100, 0)
            label = tkr.decode(label_pad, skip_special_tokens=True)
            gt_txt_test.write(label + '\n')

    for txt in [gt_txt_test]:
        txt.flush()
        txt.close()


def get_pretrained_model(model, saved_dir, log):
    saved_models = os.listdir(saved_dir)
    if len(saved_models) != 0:
        saved_models.sort()
        from_ep = saved_models[-1][5] + saved_models[-1][6] + saved_models[-1][7]
        saved_model_path = os.path.join(saved_dir, saved_models[-1])
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)
        log.info('Load state dict from %s' % str(saved_model_path))
    else:
        from_ep = -1
        log.info('Initialized randomly (with seed)')
    return model, int(from_ep)


def save_model(accelerator, model, save_path, epoch):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(),
                     os.path.join(save_path, 'epoch' + str(epoch).zfill(3) + '.pth'))
