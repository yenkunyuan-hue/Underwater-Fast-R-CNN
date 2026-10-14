class if_early_stop:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        # FIX: 原来的 best_score 只在第一次调用时赋值、之后不再更新；改为保存验证 loss 最低那一轮的权重
        self.best_state = None
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):
        # 返回 True 表示应该停止训练
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            # 拷到 CPU，避免后续训练改动这份权重，也不额外占显存
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def load_best(self, model):
        # FIX: 早停时手上的是"连续 patience 轮没变好"之后的权重，测试前换回最优权重
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
