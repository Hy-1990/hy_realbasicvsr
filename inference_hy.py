#!/user/bin/env python
# coding=utf-8
"""
@project : hy_realbasicvsr
@author  : huyi
@file   : inference_hy.py
@ide    : PyCharm
@time   : 2022-05-08 15:18:15
"""
import argparse
import glob
import os

import cv2
import mmcv
import numpy as np
import torch
import uuid
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img

from realbasicvsr.models.builder import build_model


def init_model(config, checkpoint=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.eval()

    return model


class Worker:
    def __init__(self):
        self.checkpoint_path = 'checkpoints/RealBasicVSR_x4.pth'
        self.config = 'configs/realbasicvsr_x4.py'
        self.is_save_as_png = True
        self.max_seq_len = 2
        self.model = init_model(self.config, self.checkpoint_path)

    def do_pic(self, input_image_path: str, output_dir: str):
        inputs = []
        img = mmcv.imread(input_image_path, channel_order='rgb')
        ext = os.path.basename(input_image_path).split('.')[-1]
        inputs.append(img)
        inputs.append(img)
        for i, img in enumerate(inputs):
            img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
            inputs[i] = img.unsqueeze(0)
        inputs = torch.stack(inputs, dim=1)
        # map to cuda, if available
        cuda_flag = False
        if torch.cuda.is_available():
            model = self.model.cuda()
            cuda_flag = True
        with torch.no_grad():
            if isinstance(self.max_seq_len, int):
                outputs = []
                for i in range(0, inputs.size(1), self.max_seq_len):
                    imgs = inputs[:, i:i + self.max_seq_len, :, :, :]
                    if cuda_flag:
                        imgs = imgs.cuda()
                    outputs.append(self.model(imgs, test_mode=True)['output'].cpu())
                outputs = torch.cat(outputs, dim=1)
            else:
                if cuda_flag:
                    inputs = inputs.cuda()
                outputs = self.model(inputs, test_mode=True)['output'].cpu()
        mmcv.mkdir_or_exist(output_dir)
        for i in range(0, outputs.size(1)):
            output = tensor2img(outputs[:, i, :, :, :])
            filename = '{}.{}'.format(uuid.uuid1().hex, ext)
            if self.is_save_as_png:
                file_extension = os.path.splitext(filename)[1]
                filename = filename.replace(file_extension, '.png')
            result_path = os.path.join(output_dir, filename)
            mmcv.imwrite(output, result_path)
            break


if __name__ == '__main__':
    worker = Worker()
    worker.do_pic('data/136.jpeg', 'results/')
