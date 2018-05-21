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

class ArcMarginBlock(gluon.Block):
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
      if self.margin_a>0.0:
        with self.name_scope():
          self._weight = self.params.get('fc7_weight', shape=(self.num_classes, self.emb_size))
          #if self.margin_a==0.0:
          #  self._bias = self.params.get('fc7_bias', shape=(self.num_classes, 1), lr_mult=2.0, wd_mult=0.0)
      else:
        self.dense = nn.Dense(self.num_classes, prefix='fc7')
      self.body = nn.HybridSequential(prefix='')
      feat = fresnet.get(args.num_layers, 
          version_unit=args.version_unit,
          version_act=args.version_act)
      self.body.add(feat)
      self.body.add(EmbeddingBlock(args.emb_size, args.version_output, prefix=''))

    def feature(self, x):
        feat = self.body(x)
        return feat

    def forward(self, x, label):
        feat = self.body(x)
        if self.margin_a==0.0:
          fc7 = self.dense(feat)
          #with x.context:
          #  _w = self._weight.data()
            #_b = self._bias.data()
          #fc7 = nd.FullyConnected(data=feat, weight=_w, bias = _b, num_hidden=self.num_classes, name='fc7')
          #fc7 = F.softmax_cross_entropy(data = fc7, label=label)
          return [fc7,label]

        with x.context:
          _w = self._weight.data()

        nx = nd.L2Normalization(feat, mode='instance', name='fc1n')*self.margin_s
        w = nd.L2Normalization(_w, mode='instance')
        fc7 = nd.FullyConnected(nx, w, no_bias = True, num_hidden=self.num_classes, name='fc7')
        #fc7 = self.dense(nx)
        if self.margin_a!=1.0 or self.margin_m!=0.0 or self.margin_b!=0.0:
          if self.margin_a==1.0 and self.margin_m==0.0:
            s_m = s*self.margin_b
            gt_one_hot = nd.one_hot(label, depth = self.num_classes, on_value = s_m, off_value = 0.0)
            fc7 = fc7-gt_one_hot
          else:
            zy = nd.pick(fc7, label, axis=1)
            cos_t = zy/self.margin_s
            t = nd.arccos(cos_t)
            if self.margin_a!=1.0:
              t = t*self.margin_a
            if self.margin_m>0.0:
              t = t+self.margin_m
            body = nd.cos(t)
            if self.margin_b>0.0:
              body = body - self.margin_b
            new_zy = body*self.margin_s
            diff = new_zy - zy
            diff = nd.expand_dims(diff, 1)
            gt_one_hot = nd.one_hot(label, depth = self.num_classes, on_value = 1.0, off_value = 0.0)
            body = nd.broadcast_mul(gt_one_hot, diff)
            fc7 = fc7+body
        return [fc7,label]

    #def hybrid_forward(self, F, x):
    #  feat = self.body(x)
    #  return feat

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

