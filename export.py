import argparse
import h5py
import os
import re


def save_layer(layer, names, output_dir):
    for name in names:
        weights = layer[name]
        output_file_name = name.split(':')[0].replace('/', '_')
        with open("{}/{}".format(output_dir, output_file_name), 'wb') as f:
            conf_kernel_pattern = re.compile(r"conv2d_\d_kernel")
            if conf_kernel_pattern.match(output_file_name) is not None:
                weights[()].transpose([3, 0, 1, 2]).tofile(f)
            else:
                weights[()].tofile(f)
            print("Weight `{}` saved as `{}`.".format(name, output_file_name))

def main(conf):
    f = h5py.File(conf.model_path, mode='r')
    
    if not os.path.exists(conf.output_dir):
        os.mkdir(conf.output_dir)

    weights = f['model_weights']
    layer_names = [n.decode('utf-8') for n in weights.attrs['layer_names']]
    for name in layer_names:
        layer = weights[name]
        if layer.attrs['weight_names'].size > 0:
            print("Saving layer `{}`.".format(name))
            weights_names = [name.decode('utf-8') for name in layer.attrs['weight_names']]
            save_layer(layer, weights_names, conf.output_dir)
        else:
            print("Skip layer `{}`.".format(name))
    print("FINISH")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", dest="model_path", default="model.h5",
                        help="Path of model to export.")
    parser.add_argument("--output-dir", dest="output_dir", default="weights",
                        help="Output dir of weights.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    conf = parser.parse_args()
    main(conf)
