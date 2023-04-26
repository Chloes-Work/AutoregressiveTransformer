import torch.optim as optim

class NoamScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_warmup_steps, lr_initial, model_size=None, last_epoch=-1, verbose=False):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial
        self.losses = []
        self.n_steps = 0
        self.normalize = n_warmup_steps ** 0.5
        if model_size is not None:
            self.normalize = model_size ** (-0.5)
        super().__init__(optimizer, last_epoch, verbose)
     

    def get_lr(self):
        self.n_steps += 1

        current_lr = self.optimizer.param_groups[0]["lr"]

        lr = self.lr_initial * self.get_lr_factor()

        # Changing the learning rate within the optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        if self.verbose:
            print(f"Step: {self.n_steps}, Learning Rate: {lr}")
        return [lr for _ in self.optimizer.param_groups]

    def get_lr_factor(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )