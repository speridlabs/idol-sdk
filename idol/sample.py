import argparse
import json
import os

import torch

from .conditioning import EncoderAccessError
from .pipeline import IdolPipeline

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True, help='an image, or a folder of images')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic',
                        help="'synthetic': renders and generated images of one object (crop 1.2); "
                             "'real': photos (crop 1.7)")
    parser.add_argument('--crop_margin', type=float, default=None,
                        help='override the crop: square side = object bounding box x this')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg_strength', type=float, default=3.0, help='classifier-free guidance strength')
    parser.add_argument('--rescale_t', type=float, default=3.0, help='time-step shift towards high noise')
    parser.add_argument('--num_gaussians', type=int, default=262144,
                        help='32,768 to 1,048,576, a multiple of 32')
    parser.add_argument('--weights', default='speridlabs/idol', help='Hugging Face repo or local folder')
    parser.add_argument('--device', default='cuda',
                        help="'cuda' (default), 'rocm' (or 'amd') for AMD GPUs, 'cpu', or any torch device string "
                             "(e.g. 'cuda:1')")
    parser.add_argument('--dinov3_dtype', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--no_optimize', dest='optimize', action='store_false',
                        help='turn off the faster sampling (torch.compile, batched guidance, fused attention, '
                             'TF32) that is on by default on GPU')
    args = parser.parse_args()

    if os.path.isdir(args.image):
        images = sorted(os.path.join(args.image, name) for name in os.listdir(args.image)
                        if name.lower().endswith(IMAGE_EXTENSIONS))
    else:
        images = [args.image]
    os.makedirs(args.out, exist_ok=True)

    try:
        pipeline = IdolPipeline.from_pretrained(args.weights, dinov3_dtype=args.dinov3_dtype, optimize=args.optimize,
                                                device=args.device)
    except EncoderAccessError as error:
        raise SystemExit(f'error: {error}')
    if pipeline.device.type == 'cuda':
        print(f'running on {torch.cuda.get_device_name(pipeline.device)} ({pipeline.device})')
    else:
        print('running on the CPU (very slow)')
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        output = pipeline(path, mode=args.mode, crop_margin=args.crop_margin, seed=args.seed, steps=args.steps,
                          cfg_strength=args.cfg_strength, rescale_t=args.rescale_t,
                          num_gaussians=args.num_gaussians, verbose=True)
        output.gaussian.save_ply(os.path.join(args.out, f'{name}.ply'))
        output.preprocessed_image.save(os.path.join(args.out, f'{name}_input.png'))
        with open(os.path.join(args.out, f'{name}_camera.json'), 'w') as f:
            json.dump(output.camera, f, indent=2)
        print(f'{name}: {args.out}/{name}.ply')


if __name__ == '__main__':
    main()
