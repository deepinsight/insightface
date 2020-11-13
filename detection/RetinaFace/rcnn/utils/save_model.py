import mxnet as mx


def save_checkpoint(prefix, epoch, arg_params, aux_params):
    """Checkpoint the model data into file.
    :param prefix: Prefix of model name.
    :param epoch: The epoch number of the model.
    :param arg_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    :param aux_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    :return: None
    prefix-epoch.params will be saved for parameters.
    """
    save_dict = {('arg:%s' % k): v for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v for k, v in aux_params.items()})
    param_name = '%s-%04d.params' % (prefix, epoch)
    mx.nd.save(param_name, save_dict)
