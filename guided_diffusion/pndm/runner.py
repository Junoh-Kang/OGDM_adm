# Copyright 2022 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import torch as th
import torch.distributed as dist
import numpy as np
# import torch.optim as optimi
# import torch.utils.data as data
import torchvision.utils as tvu
from scipy import integrate
# from torchdiffeq import odeint
from tqdm.auto import tqdm

# from dataset import get_dataset, inverse_data_transform
# from model.ema import EMAHelper


def get_optim(params, config):
    if config['optimizer'] == 'adam':
        optim = optimi.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'],
                            betas=(config['beta1'], 0.999), amsgrad=config['amsgrad'],
                            eps=config['eps'])
    elif config['optimizer'] == 'sgd':
        optim = optimi.SGD(params, lr=config['lr'], momentum=0.9)
    else:
        optim = None

    return optim


class Runner(object):
    def __init__(
        self,
        schedule,
        model,
        diffusion_step,
        sample_speed,
        size,
        # total_num,
        device
    ):
        self.diffusion_step = diffusion_step
        self.sample_speed = sample_speed
        self.device = device
        self.size = size
        # self.total_num = total_num

        self.schedule = schedule
        self.model = model

    def sample_fid(self, noise=None):
        # pflow = True if self.args.method == 'PF' else False
        model = self.model
        device = self.device
        pflow = False
        model.eval()

        n = self.size[0]
        # total_num = self.total_num

        skip = self.diffusion_step // self.sample_speed
        # fix this part
        seq = range(0, self.diffusion_step-1, skip)
        seq_next = [-1] + list(seq[:-1])
        image_num = 0
        if noise is None:
            noise = th.randn(*self.size, device=self.device)
        return self.sample_image(noise, seq, model, pflow)
        # if dist.get_rank() == 0:
        #     my_iter = tqdm(range(total_num // n + 1), ncols=120)
        # else:
        #     my_iter = range(total_num // n + 1)

        # for _ in my_iter:
        #     noise = th.randn(*self.size, device=self.device)

        #     img = self.sample_image(noise, seq, model, pflow)

        #     img = inverse_data_transform(config, img)
        #     for i in range(img.shape[0]):
        #         if image_num+i > total_num:
        #             break
        #         tvu.save_image(img[i], os.path.join(f"./{mpi_rank}-{image_num+i}.png"))

        #     image_num += n

    def sample_image(self, noise, seq, model, pflow=False):
        with th.no_grad():
            if pflow:
                shape = noise.shape
                device = self.device
                tol = 1e-5 if self.sample_speed > 1 else self.sample_speed

                def drift_func(t, x):
                    x = th.from_numpy(x.reshape(shape)).to(device).type(th.float32)
                    drift = self.schedule.denoising(x, None, t, model, pflow=pflow)
                    drift = drift.cpu().numpy().reshape((-1,))
                    return drift

                solution = integrate.solve_ivp(drift_func, (1, 1e-3), noise.cpu().numpy().reshape((-1,)),
                                               rtol=tol, atol=tol, method='RK45')
                img = th.tensor(solution.y[:, -1]).reshape(shape).type(th.float32)

            else:
                imgs = [noise]
                seq_next = [-1] + list(seq[:-1])

                start = True
                n = noise.shape[0]

                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (th.ones(n) * i).to(self.device)
                    t_next = (th.ones(n) * j).to(self.device)

                    img_t = imgs[-1].to(self.device)
                    img_next = self.schedule.denoising(img_t, t_next, t, model, start, pflow)
                    start = False

                    imgs.append(img_next)

                img = imgs[-1]
            return img
