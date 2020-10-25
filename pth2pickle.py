import argparse
import pickle
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
                    help='.pth file to be converted to .pickle')
args = parser.parse_args()


checkpoint = torch.load(args.checkpoint, map_location='cpu')

params = {}
for name, w in checkpoint['model'].items():
    name = name.replace('backbone.0.body', 'backbone')
    name = name.replace('bbox_embed.layers.', 'bbox_embed_')
    name = name.replace('downsample.', 'downsample_')
    name = name.replace('encoder.layers.', 'encoder.layer_')
    name = name.replace('decoder.layers.', 'decoder.layer_')
    name = name.replace('in_proj_weight', 'in_proj_kernel')
    name = name.replace('out_proj.weight', 'out_proj_kernel')
    name = name.replace('out_proj.bias', 'out_proj_bias')
    name = name.replace('.', '/')

    if 'norm' in name:
        name = name.replace('/weight', '/gamma')
        name = name.replace('/bias', '/beta')

    if 'bn' not in name and 'downsample_1' not in name:
        name = name.replace('/weight', '/kernel')

    w = w.data.cpu().numpy()

    # Adjust for the different order in which PyTorch and TF
    # represent convolutional weights
    if w.ndim == 4:
        w = w.transpose((2, 3, 1, 0))

    print('converting var:', name, w.shape)
    params[name] = w

output_file = args.checkpoint[:-3] + 'pickle'
with open(output_file, 'wb') as f:
    pickle.dump(params, f)
