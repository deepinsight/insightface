"""Compare two already-cropped body images using normalized FP32 features."""
import argparse

from insightface.app import PersonAnalysis
from demo import read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first')
    parser.add_argument('second')
    parser.add_argument('--model', choices=('cheetah_s', 'cheetah_l'), default='cheetah_s')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu else ['CPUExecutionProvider']
    with PersonAnalysis(name=args.model, providers=providers) as app:
        first = app.get_reid(read_image(args.first))
        second = app.get_reid(read_image(args.second))
        print('Body cosine similarity: %.6f' % float(first @ second))


if __name__ == '__main__':
    main()
