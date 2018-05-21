
class EmbeddingBlock(HybridBlock):
    def __init__(self, emb_size = 512, mode='E', **kwargs):
        super(EmbeddingBlock, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        if mode=='D':
          self.body.add(nn.BatchNorm())
          self.body.add(nn.Activation('relu'))
          self.body.add(nn.GlobalAvgPool2D())
          self.body.add(nn.Flatten())
          self.body.add(nn.Dense(emb_size))
          self.body.add(nn.BatchNorm(scale=False, prefix='fc1'))
        elif mode=='E':
          self.body.add(nn.BatchNorm())
          self.body.add(nn.Dropout(0.4))
          self.body.add(nn.Dense(emb_size))
          self.body.add(nn.BatchNorm(scale=False, prefix='fc1'))
        else:
          self.body.add(nn.BatchNorm())
          self.body.add(nn.Activation('relu'))
          self.body.add(nn.GlobalAvgPool2D())
          self.body.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return x

class MarginBlock(HybridBlock):
    def __init__(self, args, **kwargs):
