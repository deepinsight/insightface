from ptflops import get_model_complexity_info
from backbones import get_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('n', type=str, default="r100")
    args = parser.parse_args()
    net = get_model(args.n)
    macs, params = get_model_complexity_info(
        net, (3, 112, 112), as_strings=False,
        print_per_layer_stat=True, verbose=True)
    gmacs = macs / (1000**3)
    print("%.3f GFLOPs"%gmacs)
    print("%.3f Mparams"%(params/(1000**2)))

    if hasattr(net, "extra_gflops"):
        print("%.3f Extra-GFLOPs"%net.extra_gflops)
        print("%.3f Total-GFLOPs"%(gmacs+net.extra_gflops))

