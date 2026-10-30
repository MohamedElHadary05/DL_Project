import argparse


def main():
    parser = argparse.ArgumentParser(description='From Pixels to Language')
    sub = parser.add_subparsers(dest='command', required=True)

    train = sub.add_parser('train')
    train.add_argument('--task', required=True, choices=['classification'])
    train.add_argument('--config', required=True)

    predict = sub.add_parser('predict')
    predict.add_argument('--task', required=True, choices=['classification'])
    predict.add_argument('--image', required=True)
    predict.add_argument('--checkpoint', required=False)

    args = parser.parse_args()
    print(args)  # TODO: dispatch to training / inference


if __name__ == '__main__':
    main()
