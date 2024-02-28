class _LayerNd(nn.Module):
    def __init__(self, kernel_initializer, activation):


        if isinstance(kernel_initializer, str):
            if kernel_initializer == 'normal':
                self.kernel_initializer = init.normal_
            elif kernel_initializer == 'kaiming':
                self.kernel_initializer = init.kaiming_normal_
            elif kernel_initializer == 'xavier':
                self.kernel_initializer = init.xavier_normal_
                self.gain = init_gain.setdefault(activation, 1)
            elif kernel_initializer == 'orthogonal':
                self.kernel_initializer = init.orthogonal_
                self.gain = init_gain.setdefault(activation, 1)
        else:
            self.kernel_initializer = kernel_initializer