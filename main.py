import argparse
import functools
import logging
import datetime

import os
import sys

from runner import train


def default_parser():
    parser = argparse.ArgumentParser(description='CMCGAN')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', default=[128, 128])
    parser.add_argument('--audio_size', default=[128, 128])
    parser.add_argument('--data_path', default='.')
    parser.add_argument('--image_split_train', default='./data/image_train.txt')
    parser.add_argument('--image_split_test', default='./data/image_val.txt')
    parser.add_argument('--audio_split_train', default='./data/chunk_train.txt')
    parser.add_argument('--audio_split_test', default='./data/chunk_val.txt')
    parser.add_argument('--lambda', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--filter_base', type=int, default=64)
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--random_flip', type=float, default=0.5)
    parser.add_argument('--repeat_nums', type=int, default=5)
    parser.add_argument('--save_path', default='./save')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--disp_freq', type=int, default=20)
    parser.add_argument('--ckpt_path', default='./ckpt')
    parser.add_argument('--gpus', default='7')
    return parser.parse_args()


@functools.lru_cache()
def setup_logger(file_name, rank=0):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if rank > 0:
        return logger
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def main():
    cfg = default_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus

    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(cfg.ckpt_path, exist_ok=True)

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d_%H:%M:%S')
    file_name = os.path.join(cfg.save_path, "log_{}.txt".format(start_time))
    logger = setup_logger(file_name)

    if cfg.mode == 'train':
        logger.info('>>> Start Training ...')
        train(cfg, logger)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
