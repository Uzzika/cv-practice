import argparse

from filters import create_filter
from utils import load_image, show_images


def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description='Filters for images')
    parser.add_argument('-img', '--image_path',
                        type=str,
                        required=True,
                        help='The path to the image')

    parser.add_argument(
        '-f', '--filter',
        type=str,
        required=True,
        choices=['resize', 'sepia', 'vignette', 'pixelate', 'pixelate_interactive', 'simple_border', 'border', 'flare', 'paper'],
        help='Тype of filter'
    )

    parser.add_argument(
        '--width',
        type=int,
        help='New image width in pixels'
    )
    parser.add_argument(
        '--height',
        type=int,
        help='New image height in pixels'
    )
    parser.add_argument(
        '--scale',
        type=float,
        help='Scaling factor'
    )

    parser.add_argument(
        '--intensity',
        type=float,
        default=1,
        help='Intensity of sepia'
    )

    parser.add_argument(
        '--strength',
        type=float,
        default=0.8,
        help='Strength of vignette darkness'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=0.8,
        help='Radius of bright area in vignette'
    )

    parser.add_argument(
        '--x',
        type=int,
        default=0,
        help='X coordinate of top-left corner of pixelation region'
    )
    parser.add_argument(
        '--y',
        type=int,
        default=0,
        help='Y coordinate of top-left corner of pixelation region'
    )
    parser.add_argument(
        '--w',
        type=int,
        help='Width of pixelation region'
    )
    parser.add_argument(
        '--h',
        type=int,
        help='Height of pixelation region'
    )
    parser.add_argument(
        '--block',
        type=int,
        default=16,
        help='Pixelation block size'
    )

    parser.add_argument(
        '--border',
        type=int,
        help='Border width in pixels'
    )

    parser.add_argument(
        '--border-color',
        type=int,
        nargs=3,
        metavar=('B', 'G', 'R'),
        default=(0, 0, 0),
        help='Border color in BGR'
    )

    parser.add_argument(
        '--border-id',
        type=int,
        help='Numeric ID of PNG frame to apply'
    )
    parser.add_argument(
        '--borders-dir',
        type=str,
        default='src',
        help='Directory containing border PNG files'
    )
    parser.add_argument(
        '--opacity',
        type=float,
        default=1.0,
        help='Opacity multiplier for PNG border overlay'
    )

    parser.add_argument(
        '--flares-dir',
        type=str,
        default='src',
        help='Directory containing lens flare image'
    )

    parser.add_argument(
        '--paper-path',
        type=str,
        default='src',
        help='Directory containing lens flare image'
    )

    return parser.parse_args()


def main():
    args = cli_argument_parser()

    try:
        img = load_image(args.image_path)
        result = create_filter(args.filter, img, **vars(args))
        show_images(img, result)
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    main()
