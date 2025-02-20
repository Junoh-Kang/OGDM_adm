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

import sys
import math
import torch as th
import torch.nn as nn
import numpy as np

import sys
import copy
import torch as th


def choose_method(name):
    if name == 'DDIM':
        return gen_order_1
    elif name == 'S-PNDM':
        return gen_order_2
    elif name == 'F-PNDM':
        return gen_order_4
    elif name == 'FON':
        return gen_fon
    elif name == 'PF':
        return gen_pflow
    else:
        return None


def gen_pflow(img, t, t_next, model, betas, total_step):
    n = img.shape[0]
    beta_0, beta_1 = betas[0], betas[-1]

    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # drift, diffusion -> f(x,t), g(t)
    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = - model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

    return drift


def gen_fon(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t + t_next) / 2.0, t_next]
    if len(ets) > 2:
        noise = model(img, t)
        img_next = transfer(img, t, t-1, noise, alphas_cump)
        delta1 = img_next - img
        ets.append(delta1)
        delta = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = model(img, t_list[0])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_1 = img_ - img
        ets.append(delta_1)

        img_2 = img + delta_1 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_2, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_2 = img_ - img

        img_3 = img + delta_2 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_3, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_3 = img_ - img

        img_4 = img + delta_3 * (t - t_next).view(-1, 1, 1, 1)
        noise = model(img_4, t_list[2])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_4 = img_ - img
        delta = (1 / 6.0) * (delta_1 + 2*delta_2 + 2*delta_3 + delta_4)

    img_next = img + delta * (t - t_next).view(-1, 1, 1, 1)
    return img_next


def gen_order_4(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def runge_kutta(x, t_list, model, alphas_cump, ets):
    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1])
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1])
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et


def gen_order_2(img, t, t_next, model, alphas_cump, ets):
    if len(ets) > 0:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = 0.5 * (3 * ets[-1] - ets[-2])
    else:
        noise = improved_eular(img, t, t_next, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def improved_eular(x, t, t_next, model, alphas_cump, ets):
    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = transfer(x, t, t_next, e_1, alphas_cump)

    e_2 = model(x_2, t_next)
    et = (e_1 + e_2) / 2
    # x_next = transfer(x, t, t_next, et, alphas_cump)

    return et


def gen_order_1(img, t, t_next, model, alphas_cump, ets):
    noise = model(img, t)
    ets.append(noise)
    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def transfer(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    return x_next


def transfer_dev(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long()+1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long()+1].view(-1, 1, 1, 1)

    x_start = th.sqrt(1.0 / at) * x - th.sqrt(1.0 / at - 1) * et
    x_start = x_start.clamp(-1.0, 1.0)

    x_next = x_start * th.sqrt(at_next) + th.sqrt(1 - at_next) * et

    return x_next


def get_schedule(args, config):
    if config['type'] == "quad":
        betas = (np.linspace(config['beta_start'] ** 0.5, config['beta_end'] ** 0.5, config['diffusion_step'], dtype=np.float64) ** 2)
    elif config['type'] == "linear":
        betas = np.linspace(config['beta_start'], config['beta_end'], config['diffusion_step'], dtype=np.float64)
    elif config['type'] == 'cosine':
        betas = betas_for_alpha_bar(config['diffusion_step'], lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        betas = None

    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class Schedule(object):
    def __init__(self, args, config, device):
        # device = th.device(args.device)
        betas, alphas_cump = get_schedule(args, config)

        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        self.total_step = config['diffusion_step']

        self.method = choose_method(args.method)  # add pflow
        self.ets = None

    def diffusion(self, img, t_end, t_start=0, noise=None):
        if noise is None:
            noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, img_n, t_end, t_start, model, first_step=False, pflow=False):
        if pflow:
            drift = self.method(img_n, t_start, t_end, model, self.betas, self.total_step)

            return drift
        else:
            if first_step:
                self.ets = []
            img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)

            return img_next

