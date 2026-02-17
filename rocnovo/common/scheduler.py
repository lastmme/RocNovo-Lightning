import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

def get_restart_cosine_decay_scheduler_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int=1,
    lr_decay_factor: float=5.0,
    last_epoch: int=-1,
):
    """
    Creates a scheduler that combines a linear warmup, cosine restarts, and a
    decaying peak learning rate after each restart.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (int, optional, defaults to 1):
            The number of cosine cycles (restarts) to perform.
        lr_decay_factor (float, optional, defaults to 5.0):
            The factor by which the peak learning rate is divided after each cycle.
        last_epoch (int, optional, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        torch.optim.lr_scheduler.LambdaLR: A scheduler with the specified behavior.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress_after_warmup = float(current_step - num_warmup_steps)
        steps_per_cycle = float(max(1, num_training_steps - num_warmup_steps)) / float(max(1, num_cycles))

        current_cycle = math.floor(progress_after_warmup / steps_per_cycle)
        peak_lr_scale = 1.0 / (lr_decay_factor ** current_cycle)
        progress_in_cycle = (progress_after_warmup % steps_per_cycle) / steps_per_cycle
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress_in_cycle))

        return max(0.0, peak_lr_scale * cosine_scale)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)