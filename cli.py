import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    args = parser.parse_args()
    train(render=args.render)
