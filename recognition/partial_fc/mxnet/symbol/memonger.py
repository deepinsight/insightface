import mxnet as mx
import math

def prod(shape):
    """Get product of the shape.
    """
    ret = 1
    for s in shape:
        ret *= s
    return ret


def is_param(name):
    """Quick script to check if name is a parameter.
    """
    if name == 'data':
        return False
    if name.endswith('weight'):
        return True
    if name.endswith('bias'):
        return True
    if name.endswith('beta'):
        return True
    if name.endswith('gamma'):
        return True
    return False


def make_mirror_plan(sym, threshold, plan_info=None, **kwargs):
    """Memory allocation planner with a given threshold.

    The user can pass in a network configuration,
    a threshold that limits memory per block.
    And input shape configurations.

    Parameters
    ----------
    sym : symbol
        Input configuration of symbols.
        The user need to pre-mark the attribute "mirror_stage" on the nodes
        that can be book-kept as stage

        The algorithm will decide whether to disbale mirror on the stage nodes.

    threshold: integer
        A tuning parameter to tune the approximate size of each stage blocks

    plan_info: dict, optional
        Used to hold plan information.

    **kwargs:
        The arguments to infer shape.

    Returns
    -------
    alloc_sym: symbol
        A symbol with force mirror tagged on the nodes for better allocation.
    """
    threshold = threshold << 20
    sym = sym.__copy__()
    internals = sym.get_internals()
    _, out_shapes, _ = internals.infer_shape(**kwargs)
    shape_dict = list(zip(internals.list_outputs(), out_shapes))
    total_size = 0
    param_size = 0
    local_size = 0
    save_size = 0
    max_size = 0
    last_sb = None
    last_local = 0
    period = 1
    last_stage = ''
    stage_decision = ''

    for idx, item in enumerate(shape_dict):
        sb = internals[idx]
        name, shape = item
        if is_param(name):
            param_size += prod(shape) * 4
            continue
        else:
            total_size += prod(shape) * 4
            local_size += prod(shape) * 4
            sb._set_attr(force_mirroring='True')

        if sb.attr('mirror_stage') is not None:
            stage = sb.attr('mirror_stage')
            if stage == 'True' or stage != last_stage:
                if local_size > threshold:
                    save_size += prod(shape) * 4
                    max_size = max(max_size, local_size)
                    local_size = 0
                    stage_decision = 'False'
                    sb._set_attr(force_mirroring=stage_decision)
                else:
                    stage_decision = 'True'
                    pass
                last_stage = stage
            elif stage == last_stage and stage_decision == 'False':
                save_size += prod(shape) * 4
                sb._set_attr(force_mirroring=stage_decision)

    if plan_info is not None:
        plan_info['max_size'] = max_size
        plan_info['save_size'] = save_size
    return sym


def get_cost(sym, type_dict=None, **kwargs):
    """Get the cost of the current symbolic plan by running bind on CPU.

    sym : Symbolic Variable

    """
    texec = sym.simple_bind(ctx=mx.gpu(),
                            grad_req='write',
                            type_dict=type_dict,
                            **kwargs)
    return int(texec.debug_str().split('\n')[-3].split()[1])


def search_plan(sym, ntrial=6, type_dict=None, **kwargs):
    """Quickly heurestic search over possible plans to find good memory plan.

    Parameters
    ----------
    sym : symbolic
       Symbolic configurations

    ntrial: integer
       Additional grid search steps
    """
    history = []
    threshold = 0
    min_threshold = None
    min_cost = None
    nbegin = 3

    for k in range(nbegin):
        info = {}
        sym = make_mirror_plan(sym, threshold=threshold, plan_info=info, **kwargs)
        cost = get_cost(sym, type_dict, **kwargs)
        save_size = info['save_size'] >> 20
        local_size = info['max_size'] >> 20
        guess = int(math.sqrt(save_size * local_size / 2))
        if min_cost is None or min_cost > cost:
            min_cost = cost
        if min_threshold is None or local_size < min_threshold:
            min_threshold = local_size
        print ("Search threshold=%d MB, cost=%d MB" % (threshold, cost))
        history.append((cost, threshold, sym))
        threshold = guess

    max_threshold = threshold * math.sqrt(2)
    step = int((max_threshold - min_threshold) / ntrial)
    threshold = min_threshold + step
    if step > 0:
        for k in range(ntrial):
            sym = make_mirror_plan(sym, threshold=threshold, plan_info=info, **kwargs)
            cost = get_cost(sym, type_dict, **kwargs)
            print ("Search threshold=%d MB, cost=%d MB" % (threshold, cost))
            history.append((cost, threshold, sym))
            threshold += step

    history.sort(key = lambda x: x[0])
    cost, threshold, sym = history[0]
    print('Find best plan with threshold=%d, cost=%d MB' % (threshold, cost))
    return sym
