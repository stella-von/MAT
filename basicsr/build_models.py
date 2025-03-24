import warnings

from basicsr.archs.mat_arch import MAT

warnings.filterwarnings("ignore", category=UserWarning)


def build_models(model_name, resolution='720p', upscale=4):
    model = None

    if resolution.lower() == '540p':
        hr_height, hr_width = 960, 540
    elif resolution.lower() == '720p':
        hr_height, hr_width = 1280, 720
    elif resolution.lower() == '1080p':
        hr_height, hr_width = 1920, 1080
    elif resolution.lower() == '2k':
        hr_height, hr_width = 2560, 1440
    elif resolution.lower() == '4k':
        hr_height, hr_width = 3840, 2160
    elif resolution.isdigit():
        hr_height, hr_width = int(resolution), int(resolution)

    if 'mat' in model_name.lower():
        lr_height, lr_width = hr_height // upscale, hr_width // upscale

        if 'light' in model_name.lower():
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
                upscale=upscale,
                upsampler='pixelshuffledirect')
        elif model_name.lower() == 'mat':
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
                dw_size=[1, 3, 5, 7],
                upscale=upscale,
                upsampler='pixelshuffle')

    else:
        raise ValueError(f'model {model_name} is not supported.')

    return model, (lr_height, lr_width)
