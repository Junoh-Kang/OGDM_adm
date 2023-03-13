from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    # name = name.replace("'", "").replace('"','')
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss_aware":
        return LossSecondMomentResampler(diffusion)
    elif name.startswith("pair_T"):
        return eval("PairSampler_T(diffusion, ratio=" + name.split(",")[-1] + ")")
    elif name.startswith("pair_t"):
        return eval("PairSampler_t(diffusion, ratio=" + name.split(",")[-1] + ")")
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class _UniformSampler(ScheduleSampler):
    def __init__(self, T):
        self._weights = np.ones([T])

    def weights(self):
        return self._weights

class PairSampler_T():
    def __init__(self, diffusion, ratio=0):
        self.diffusion = diffusion
        self.ratio = ratio

    def sample(self, batch_size, device):
        sampler = _UniformSampler(self.diffusion.num_timesteps)
        ts, weights = sampler.sample(batch_size, device)
        s = []
        for t in ts.cpu().numpy():
            s_max = min(int(self.ratio * self.diffusion.num_timesteps), t) + 1
            tmp, _ = _UniformSampler(s_max).sample(1, device)
            s.append(tmp)
        s = th.cat(s).long().to(device)
        return ts, weights, s

class PairSampler_t():
    def __init__(self, diffusion, ratio=0):
        self.diffusion = diffusion
        self.ratio = ratio

    def sample(self, batch_size, device):
        sampler = _UniformSampler(self.diffusion.num_timesteps)
        ts, weights = sampler.sample(batch_size, device)
        s = []
        for t in ts.cpu().numpy():
            s_max = int(self.ratio * t) + 1
            tmp, _ = _UniformSampler(s_max).sample(1, device)
            s.append(tmp)
        s = th.cat(s).long().to(device)
        return ts, weights, s
# class DiscAwareResampler(ScheduleSampler):
#     def __init__(self, diffusion, sample_type="uniform", 
#                        loss_name="", loss_target=0.7, ):        
        
#         # self.diffusion = diffusion
#         self.T_max = diffusion.num_timesteps
#         self.T_min = self.T_max // 100
#         self.T_step = self.T_max // 100
#         self.T_cur = self.T_max
        
#         self.loss_target = loss_target
#         self.loss_cur = 0
#         self.loss_ema = 0
        
#         self.ema_rate = 0.9

#         self.sample_type = sample_type
#         self._weights = self.get_weights()

#     def weights(self):
#         return self._weights

#     def get_weights(self):
#         if self.sample_type == "uniform":
#             return np.concatenate((np.ones(self.T_cur), np.zeros(self.T_max-self.T_cur)))[::-1]
#         elif self.sample_type == "priority":
#             return np.concatenate((np.arange(self.T_cur), np.zeros(self.T_max-self.T_cur)))[::-1]

#     def update_with_local_losses(self, losses):
#         # update loss
#         self.loss_cur = losses.mean().squeeze()
#         self.loss_ema = self.ema_rate * self.loss_cur + (1 - self.ema_rate) * self.loss_ema
#         # update T_cur and weights
        
#         # Discriminator가 잘하면 (ACC 기준) T_cur을 증가
#         if self.loss_ema > self.loss_target: 
#             self.T_cur = min(self.T_cur + self.T_step, self.T_max)
#         # Discriminator가 못하면 (ACC 기준) T_cur을 감소
#         else: 
#             self.T_cur = max(self.T_cur - self.T_step, self.T_min)
#         self._weights = self.get_weights()
#         return self.T_cur

class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()