import mxnet as mx


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
      if 'bbox_pred_weight' in arg:
        arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
        arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
      mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
      if 'bbox_pred_weight' in arg:
        arg.pop('bbox_pred_weight_test')
        arg.pop('bbox_pred_bias_test')
    return _callback
