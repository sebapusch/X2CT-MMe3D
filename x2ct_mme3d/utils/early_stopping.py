class EarlyStopping:
    def __init__(self, patience: int, gt_is_better: bool = False):
        self.patience = patience
        self.prev_best = None
        self.counter = 0
        self.gt_is_better = gt_is_better

    def _is_better(self, current: float, prev: float) -> bool:
        if self.gt_is_better: return current > prev
        return current < prev

    def reset(self):
        self.prev_best = None
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        if self.prev_best is None:
            self.prev_best = metric
            return False

        if self._is_better(metric, self.prev_best):
            self.prev_best = metric
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.reset()
            return True

        return False



