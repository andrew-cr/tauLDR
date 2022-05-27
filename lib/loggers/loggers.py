import ml_collections
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
from pathlib import Path
import os
import time
from tqdm import tqdm

import lib.loggers.logger_utils as logger_utils
import lib.sampling.sampling_utils as sampling_utils
import lib.sampling.sampling as sampling
import lib.losses.losses as losses
import lib.utils.bookkeeping as bookkeeping


@logger_utils.register_logger
def denoisingImages(*args, **kwargs):
    state = kwargs['state']
    cfg = kwargs['cfg']
    writer = kwargs['writer']
    minibatch = kwargs['minibatch']
    dataset = kwargs['dataset']
    model = state['model']

    ts = [0.01, 0.3, 0.5, 0.6,0.7,0.8, 1.0]
    C,H,W = cfg.data.shape
    B = 1
    S = cfg.data.S

    def imgtrans(x):
        # C,H,W -> H,W,C
        x = x.transpose(0,1)
        x = x.transpose(1,2)
        return x

    fig, ax = plt.subplots(6, len(ts))
    for img_idx in range(3):
        for t_idx in range(len(ts)):
            qt0 = model.transition(torch.tensor([ts[t_idx]], device=model.device)) # (B, S, S)
            qt0_rows = qt0[
                0, minibatch[img_idx, ...].flatten().long(), :
            ]
            x_t_cat = torch.distributions.categorical.Categorical(
                qt0_rows
            )
            x_t = x_t_cat.sample().view(1, C*H*W)

            x_0_logits = model(x_t, torch.tensor([ts[t_idx]], device=model.device)).view(B,C,H,W,S)
            x_0_max_logits = torch.max(x_0_logits, dim=4)[1]

            ax[2*img_idx, t_idx].imshow(imgtrans(x_t.view(B,C,H,W)[0, ...].detach().cpu()))
            ax[2*img_idx, t_idx].axis('off')
            ax[2*img_idx+1, t_idx].imshow(imgtrans(x_0_max_logits[0, ...].detach().cpu()))
            ax[2*img_idx+1, t_idx].axis('off')

    writer.add_figure('denoisingImages', fig, state['n_iter'])


@logger_utils.register_logger
def ConditionalDenoisingNoteSeq(*args, **kwargs):
    state = kwargs['state']
    cfg = kwargs['cfg']
    writer = kwargs['writer']
    dataset = kwargs['dataset']
    model = state['model']
    minibatch = kwargs['minibatch']

    ts = [0.01, 0.1, 0.3, 0.7, 1.0]
    total_L = cfg.data.shape[0]
    data_L = cfg.data.shape[0] - cfg.loss.condition_dim
    S = cfg.data.S


    with torch.no_grad():
        fig, ax = plt.subplots(2, len(ts))
        for data_idx in range(1):
            for t_idx in range(len(ts)):
                qt0 = model.transition(torch.tensor([ts[t_idx]], device=model.device)) # (B, S, S)
                conditioner = minibatch[data_idx, 0:cfg.loss.condition_dim].view(1, cfg.loss.condition_dim)
                data = minibatch[data_idx, cfg.loss.condition_dim:].view(1, data_L)
                qt0_rows = qt0[
                    0, data.flatten().long(), :
                ]
                x_t_cat = torch.distributions.categorical.Categorical(
                    qt0_rows
                )
                x_t = x_t_cat.sample().view(1, data_L)

                model_input = torch.concat((conditioner, x_t), dim=1)

                full_x_0_logits = model(model_input, torch.tensor([ts[t_idx]], device=model.device)).view(1,total_L,S)
                x_0_logits = full_x_0_logits[:, cfg.loss.condition_dim:, :]

                x_0_max_logits = torch.max(x_0_logits, dim=2)[1]

                x_0_np = x_0_max_logits.cpu().detach().numpy()
                x_t_np = x_t.cpu().detach().numpy()
                conditioner_np = conditioner[data_idx, :].cpu().detach().numpy()

                ax[2*data_idx, t_idx].scatter(np.arange(total_L),
                    np.concatenate((conditioner_np, x_t_np[0, :]), axis=0), s=0.1)
                ax[2*data_idx, t_idx].axis('off')
                ax[2*data_idx, t_idx].set_ylim(0, S)
                ax[2*data_idx+1, t_idx].scatter(np.arange(total_L),
                    np.concatenate((conditioner_np, x_0_np[0, :]), axis=0), s=0.1)
                ax[2*data_idx+1, t_idx].axis('off')
                ax[2*data_idx+1, t_idx].set_ylim(0, S)

    fig.set_size_inches(len(ts)*2, 2*2)

    writer.add_figure('ConditionaldenoisingNoteSeq', fig, state['n_iter'])