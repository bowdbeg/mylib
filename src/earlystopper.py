from copy import deepcopy


class EarlyStopper():
    def __init__(self,
                 model,
                 patience=10,
                 threshold=1e-3,
                 startup=0,
                 mode='min'):
        assert mode in ['min', 'max']
        self.patience = patience
        self.threshold = threshold
        self.startup = startup
        self.values = []
        self.mode = mode
        if self.mode == 'min':
            self.sign = 1.
        elif self.mode == 'max':
            self.sign = -1.
        self.count = 0
        self.model = model
        self.best_param = deepcopy(model.state_dict())
        self.best_epoch = 0

    def step(self, val):
        if len(self.values) == 0:
            self.values.append(val)
            return False
        val_past = self.values[-1]
        sub = val - val_past
        self.values.append(val)
        if self.sign * sub > 0 or abs(sub) < self.threshold:
            self.count += 1
        else:
            self.count = 0
            self.best_param = deepcopy(self.model.state_dict())
            self.best_epoch = len(self.values)

        return self.count > self.patience and len(self.values) > self.startup