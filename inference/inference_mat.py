import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.mat_arch import MAT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lightweight_sr', help='classical_sr, lightweight_sr')
    parser.add_argument('--input', type=str, default='datasets/test', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/MAT/test', help='output folder')
    parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
    parser.add_argument('--upscale', type=int, default=4, help='scale factor: 2, 3, 4')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/MAT_light_x4.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_MAT.png'), output)


def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = MAT(
            num_feat=156,
            num_block=6,
            expanded_ratio=2,
            squeeze_factor=2,
            depth=6,
            num_head=6,
            kernel_sizes=[13, 15, 17],
            dilations=[[1, 1, 1], [4, 4, 3]],
            rel_pos_bias=True,
            dw_sizes=[1, 3, 5, 7],
            upscale=args.upscale,
            upsampler='pixelshuffle')

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = MAT(
            num_feat=60,
            num_block=4,
            expanded_ratio=1,
            squeeze_factor=4,
            depth=4,
            num_head=6,
            kernel_sizes=[7, 9, 11],
            dilations=[[1, 1, 1], [9, 7, 5]],
            rel_pos_bias=True,
            mlp_ratio=2,
            dw_sizes=[1, 3, 5, 7],
            upscale=args.upscale,
            upsampler='pixelshuffledirect')

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
