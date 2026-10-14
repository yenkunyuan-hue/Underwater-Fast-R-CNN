class if_early_stop:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_score = None

    def __call__(self, val_loss, best_score):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_score = best_score
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop, self.best_loss, self.best_score
