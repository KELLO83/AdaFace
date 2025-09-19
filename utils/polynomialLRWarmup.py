from torch.optim.lr_scheduler import _LRScheduler
import warnings

class PolynomialLRWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters=5, power=1.0, last_epoch=-1 , limit_lr=1e-5):
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters
        self.limit_lr = limit_lr  # 최소 학습률 제한

        super().__init__(optimizer, last_epoch=last_epoch )


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.warmup_iters > 0 and self.last_epoch <= self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        elif self.total_iters == self.warmup_iters:
            return [max(self.limit_lr, group["lr"]) if self.limit_lr is not None else group["lr"] for group in self.optimizer.param_groups]
        else:        
            l = self.last_epoch
            w = self.warmup_iters
            t = self.total_iters
            decay_factor = ((1.0 - (l - w) / (t - w)) / (1.0 - (l - 1 - w) / (t - w))) ** self.power

        new_lrs = []
        for group in self.optimizer.param_groups:
            updated_lr = group["lr"] * decay_factor
            if self.limit_lr is not None:
                updated_lr = max(updated_lr, self.limit_lr)
            new_lrs.append(updated_lr)
        return new_lrs

    def _get_closed_form_lr(self):

        if self.warmup_iters > 0 and self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        elif self.total_iters == self.warmup_iters:
            return [max(self.limit_lr, base_lr) if self.limit_lr is not None else base_lr for base_lr in self.base_lrs]
        else:
            closed_form_lrs = []
            for base_lr in self.base_lrs:
                progress = (min(self.total_iters, self.last_epoch) - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                updated_lr = base_lr * (1.0 - progress) ** self.power
                if self.limit_lr is not None:
                    updated_lr = max(updated_lr, self.limit_lr)
                closed_form_lrs.append(updated_lr)
            return closed_form_lrs



from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]
