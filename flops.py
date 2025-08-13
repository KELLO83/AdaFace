from ptflops import get_model_complexity_info
import argparse
from net import Backbone

if __name__ == '__main__':
    net = Backbone([112,112], 100, 'ir')
    net = Backbone([112,112],50,'ir')
    net.eval()
    macs, params = get_model_complexity_info(
        net, (3, 112, 112), as_strings=False,
        print_per_layer_stat=True, verbose=True)
    gmacs = macs / (1000**3)
    print("%.3f GFLOPs"%gmacs)
    print("%.3f Mparams"%(params/(1000**2)))

    if hasattr(net, "extra_gflops"):
        print("%.3f Extra-GFLOPs"%net.extra_gflops)
        print("%.3f Total-GFLOPs"%(gmacs+net.extra_gflops))

