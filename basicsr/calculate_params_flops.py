import argparse
import torch
import warnings
from fvcore.nn import FlopCountAnalysis, flop_count_table
from natten.flops import add_natten_handle

from basicsr.build_models import build_models

warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    model_name = args.model_name
    resolution = args.resolution
    upscale = args.upscale
    model, (lr_height, lr_width) = build_models(model_name, resolution, upscale)

    net_cls_str = f'{model.__class__.__name__}'
    if '-' in model_name:
        net_cls_str += '-' + model_name.split('-')[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.multi_adds:
        # fvcore
        flop_ctr = FlopCountAnalysis(model, torch.randn(1, 3, lr_height, lr_width).to(device))
        if 'mat' in model_name.lower() or 'mrat' in model_name.lower():
            flop_ctr = add_natten_handle(flop_ctr)
        disable_warnings = True
        if disable_warnings:
            flop_ctr = flop_ctr.unsupported_ops_warnings(False)
        print(flop_count_table(flop_ctr))
        print(f"Network: {net_cls_str} (x{upscale}), with active parameters: {params/1e3:.02f} K," +
              f" with flops ({lr_height * upscale} x {lr_width * upscale}): {flop_ctr.total()/1e9:.2f} GMac.")
    else:
        print(f"Network: {net_cls_str} (x{upscale}), with active parameters: {params/1e3:.02f} K.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for calculating parameters and flops of a model')
    parser.add_argument('-m', '--model_name', type=str, default='mat-light', help='model name')
    parser.add_argument('-r', '--resolution', type=str, default='720p', help='resolution, progressive scanning')
    parser.add_argument('-up', '--upscale', type=int, default=4, help='upscale factor')
    parser.add_argument('-ma', '--multi_adds', type=bool, default=True, help='whether to calculate multi-adds')
    args = parser.parse_args()
    main(args)
