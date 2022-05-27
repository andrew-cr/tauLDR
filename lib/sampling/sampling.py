import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.integrate
import math
from tqdm import tqdm

import lib.sampling.sampling_utils as sampling_utils

def get_initial_samples(N, D, device, S, initial_dist, initial_dist_std=None):
    if initial_dist == 'uniform':
        x = torch.randint(low=0, high=S, size=(N, D), device=device)
    elif initial_dist == 'gaussian':
        target = np.exp(
            - ((np.arange(1, S+1) - S//2)**2) / (2 * initial_dist_std**2)
        )
        target = target / np.sum(target)

        cat = torch.distributions.categorical.Categorical(
            torch.from_numpy(target)
        )
        x = cat.sample((N*D,)).view(N,D)
        x = x.to(device)
    else:
        raise NotImplementedError('Unrecognized initial dist ' + initial_dist)
    return x


@sampling_utils.register_sampler
class TauLeaping():
    def __init__(self, cfg):
        self.cfg =cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0
        C,H,W = self.cfg.data.shape
        D = C*H*W
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)


            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                qt0 = model.transition(t * torch.ones((N,), device=device)) # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                p0t = F.softmax(model(x, t * torch.ones((N,), device=device)), dim=2) # (N, D, S)


                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())



                qt0_denom = qt0[
                    torch.arange(N, device=device).repeat_interleave(D*S),
                    torch.arange(S, device=device).repeat(N*D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N,D,S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0 # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(D*S),
                    torch.arange(S, device=device).repeat(N*D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N, D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(D),
                    torch.arange(D, device=device).repeat(N),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1,1,S) - x.view(N,D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()
                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist

@sampling_utils.register_sampler
class PCTauLeaping():
    def __init__(self, cfg):
        self.cfg =cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0

        C,H,W = self.cfg.data.shape
        D = C*H*W
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        device = model.device

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)

            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):

                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    p0t = F.softmax(model(in_x, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)


                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N,D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, D, S)

                    reverse_rates = forward_rates * ((p0t / qt0_denom) @ qt0_numer) # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*D)
                    ].view(N, D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    unclip_x_new = in_x + overall_jump
                    x_new = torch.clamp(unclip_x_new, min=0, max=S-1)

                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.detach().cpu().numpy())
                    x0_hist.append(x_0max.detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)

                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t-h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist

@sampling_utils.register_sampler
class ConditionalTauLeaping():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        reject_multiple_jumps = scfg.reject_multiple_jumps
        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                qt0 = model.transition(t * torch.ones((N,), device=device)) # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                model_input = torch.concat((conditioner, x), dim=1)
                p0t = F.softmax(model(model_input, t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                p0t = p0t[:, condition_dim:, :]


                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())



                qt0_denom = qt0[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N,sample_D,S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0 # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N, sample_D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(sample_D),
                    torch.arange(sample_D, device=device).repeat(N),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1,1,S) - x.view(N,sample_D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()

                if reject_multiple_jumps:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                    adj_diffs = masked_jump_nums * diffs
                else:
                    adj_diffs = jump_nums * diffs


                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist


@sampling_utils.register_sampler
class ConditionalPCTauLeaping():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        reject_multiple_jumps = scfg.reject_multiple_jumps
        eps_ratio = scfg.eps_ratio

        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(model(model_input, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]


                    x_0max = torch.max(p0t, dim=2)[1]


                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        x.long().flatten().repeat_interleave(S)
                    ].view(N,sample_D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, sample_D, S)

                    reverse_rates = forward_rates * ((p0t/qt0_denom) @ qt0_numer) # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*sample_D)
                    ].view(N, sample_D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,sample_D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()

                    if reject_multiple_jumps:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                        adj_diffs = masked_jump_nums * diffs
                    else:
                        adj_diffs = jump_nums * diffs

                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = in_x + overall_jump
                    x_new = torch.clamp(xp, min=0, max=S-1)
                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)
                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t-h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate,
                            corrector_step_size_multiplier * h)



            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist

