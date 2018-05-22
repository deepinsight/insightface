import mxnet as mx
from mxnet import gluon
from mxnet import profiler
from mxnet.gluon import nn
from mxnet import ndarray as nd
import fresnet

class EmbeddingBlock(gluon.HybridBlock):
    def __init__(self, emb_size = 512, mode='E', **kwargs):
        super(EmbeddingBlock, self).__init__(**kwargs)
        self.emb_size = emb_size
        print('mode', mode)
        with self.name_scope():
          self.body = nn.HybridSequential(prefix='')
          if mode=='D':
            self.body.add(nn.BatchNorm())
            self.body.add(nn.Activation('relu'))
            self.body.add(nn.GlobalAvgPool2D())
            self.body.add(nn.Flatten())
            self.body.add(nn.Dense(emb_size))
            self.body.add(nn.BatchNorm(scale=False, prefix='fc1'))
          elif mode=='E':
            self.body.add(nn.BatchNorm(epsilon=2e-5))
            self.body.add(nn.Dropout(0.4))
            #self.body.add(nn.Flatten())
            self.body.add(nn.Dense(emb_size))
            self.body.add(nn.BatchNorm(scale=False, epsilon=2e-5, prefix='fc1'))
          elif mode=='Z':
            #self.body.add(nn.BatchNorm(epsilon=2e-5))
            #self.body.add(nn.Activation('relu'))
            #self.body.add(nn.GlobalAvgPool2D())
            #self.body.add(nn.Flatten())
            self.body.add(nn.BatchNorm(epsilon=2e-5))
            self.body.add(nn.Dropout(0.4))
            #self.body.add(nn.Flatten())
            self.body.add(nn.Dense(emb_size))
            #self.body.add(nn.BatchNorm(scale=False, epsilon=2e-5, prefix='fc1'))
          else:
            self.body.add(nn.BatchNorm(epsilon=2e-5))
            self.body.add(nn.Activation('relu'))
            self.body.add(nn.GlobalAvgPool2D())
            self.body.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.body(x)
        #bn_mom = 0.9
        #x = F.BatchNorm(data=x, fix_gamma=True, eps=2e-5, momentum=bn_mom)
        return x
        #return x

class ArcMarginBlock(gluon.HybridBlock):
    def __init__(self, args, **kwargs):
      super(ArcMarginBlock, self).__init__(**kwargs)
      self.margin_s = args.margin_s
      self.margin_m = args.margin_m
      self.margin_a = args.margin_a
      self.margin_b = args.margin_b
      self.num_classes = args.num_classes
      self.emb_size = args.emb_size
      #self.weight = gluon.Parameter(name = 'fc7_weight', shape = (self.num_classes, self.emb_size))
      #self.weight.initialize()
      #self._weight = nd.empty(shape = (self.num_classes, self.emb_size))
      #if self.margin_a>0.0:
      with self.name_scope():
        self.fc7_weight = self.params.get('fc7_weight', shape=(self.num_classes, self.emb_size))
      #else:
      #  self.dense = nn.Dense(self.num_classes, prefix='fc7')
      self.body = nn.HybridSequential(prefix='')
      feat = fresnet.get(args.num_layers, 
          version_unit=args.version_unit,
          version_act=args.version_act)
      self.body.add(feat)
      self.body.add(EmbeddingBlock(args.emb_size, args.version_output, prefix=''))

    def feature(self, x):
        feat = self.body(x)
        return feat

    def hybrid_forward(self, F, x, label, fc7_weight):
        feat = self.body(x)
        if self.margin_a==0.0:
          fc7 = F.FullyConnected(feat, fc7_weight, no_bias = True, num_hidden=self.num_classes, name='fc7')
          #fc7 = self.dense(feat)
          #with x.context:
          #  _w = self._weight.data()
            #_b = self._bias.data()
          #fc7 = nd.FullyConnected(data=feat, weight=_w, bias = _b, num_hidden=self.num_classes, name='fc7')
          #fc7 = F.softmax_cross_entropy(data = fc7, label=label)
          return fc7

        nx = F.L2Normalization(feat, mode='instance', name='fc1n')*self.margin_s
        w = F.L2Normalization(fc7_weight, mode='instance')
        fc7 = F.FullyConnected(nx, w, no_bias = True, num_hidden=self.num_classes, name='fc7')
        #fc7 = self.dense(nx)
        if self.margin_a!=1.0 or self.margin_m!=0.0 or self.margin_b!=0.0:
          if self.margin_a==1.0 and self.margin_m==0.0:
            s_m = s*self.margin_b
            gt_one_hot = F.one_hot(label, depth = self.num_classes, on_value = s_m, off_value = 0.0)
            fc7 = fc7-gt_one_hot
          else:
            zy = F.pick(fc7, label, axis=1)
            cos_t = zy/self.margin_s
            t = F.arccos(cos_t)
            if self.margin_a!=1.0:
              t = t*self.margin_a
            if self.margin_m>0.0:
              t = t+self.margin_m
            body = F.cos(t)
            if self.margin_b>0.0:
              body = body - self.margin_b
            new_zy = body*self.margin_s
            diff = new_zy - zy
            diff = F.expand_dims(diff, 1)
            gt_one_hot = F.one_hot(label, depth = self.num_classes, on_value = 1.0, off_value = 0.0)
            body = F.broadcast_mul(gt_one_hot, diff)
            fc7 = fc7+body
        return fc7

    #def hybrid_forward(self, F, x):
    #  feat = self.body(x)
    #  return feat

class DenseBlock(gluon.HybridBlock):
    def __init__(self, args, **kwargs):
      super(DenseBlock, self).__init__(**kwargs)
      self.num_classes = args.num_classes
      self.emb_size = args.emb_size
      self.body = nn.HybridSequential(prefix='')
      feat = fresnet.get(args.num_layers, 
          version_unit=args.version_unit,
          version_act=args.version_act)
      self.body.add(feat)
      self.body.add(EmbeddingBlock(args.emb_size, args.version_output, prefix=''))
      self.dense = nn.Dense(self.num_classes, prefix='fc7')

    def feature(self, x):
        feat = self.body(x)
        return feat

    def hybrid_forward(self, F, x):
        feat = self.body(x)
        fc7 = self.dense(feat)
        return fc7

class ArcMarginTestBlock(gluon.Block):
    def __init__(self, args, **kwargs):
      super(ArcMarginTestBlock, self).__init__(**kwargs)

      self.body = nn.HybridSequential(prefix='')
      feat = fresnet.get(args.num_layers, 
          version_unit=args.version_unit,
          version_act=args.version_act)
      self.body.add(feat)
      self.body.add(EmbeddingBlock(args.emb_size, args.version_output))

    def forward(self, x):
      feat = self.body(x)
      return feat

class _GABlock(gluon.HybridBlock):
    def __init__(self, args, num_classes, **kwargs):
      super(_GABlock, self).__init__(**kwargs)
      with self.name_scope():
        self.body = nn.HybridSequential(prefix='')
        feat = fresnet.get(args.num_layers, 
            version_unit=args.version_unit,
            version_act=args.version_act)
        self.body.add(feat)
        self.body.add(EmbeddingBlock(mode=args.version_output))
        self.body.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
      return self.body(x)


class GABlock(gluon.HybridBlock):
    def __init__(self, args, **kwargs):
      super(GABlock, self).__init__(**kwargs)
      with self.name_scope():
        #args.num_classes = 2
        self.bodyg = _GABlock(args, 2, prefix='gender_')
        #args.num_classes = 200
        self.bodya = _GABlock(args, 200, prefix='age_')
        #if args.task=='age':
        #  self.bodyg.collect_params().setattr('grad_req', 'null')
        #elif args.task=='gender':
        #  self.bodya.collect_params().setattr('grad_req', 'null')
        #self.body = nn.HybridSequential(prefix='')

    def hybrid_forward(self, F, x):
      g = self.bodyg(x)
      a = self.bodya(x)
      f = F.concat(g,a,dim=1, name='fc1')
      return [f,g,a]

