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
    name = name.replace('.', '/')
    w = w.data.cpu().numpy()

    # Adjust for the different order in which PyTorch and TF
    # represent convolutional weights
    if w.ndim == 4:
        w = w.transpose((2, 3, 1, 0))

    if 'norm' in name:
        name = name.replace('/weight', '/gamma')
        name = name.replace('/bias', '/beta')

    if 'bn' not in name and 'downsample/1' not in name:
        name = name.replace('/weight', '/kernel')

    print('converting var:', name, w.shape)
    params[name] = w

output_file = args.checkpoint[:-3] + 'pickle'
with open(output_file, 'wb') as f:
    pickle.dump(params, f)
