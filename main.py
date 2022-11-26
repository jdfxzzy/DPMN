import yaml
import csv
import argparse
import os
from easydict import EasyDict
from interfaces.super_resolution import TextSR
from utils.util import set_seed
import warnings
warnings.filterwarnings("ignore")


def main(config, args):
    set_seed(config.TRAIN.manualSeed)
    Mission = TextSR(config, args)
    if args.test:
        if not os.path.exists(config.TRAIN.ckpt_dir):
            os.mkdir(config.TRAIN.ckpt_dir)
        result_path = os.path.join(config.TRAIN.ckpt_dir, "test_result.csv")
        if not os.path.exists(result_path):
            with open(result_path, "w+") as out:
                writer = csv.writer(out)
                writer.writerow(["recognizer", "subset", "accuracy", "psnr", "ssim"])
        Mission.test()
    else:
        if not os.path.exists(config.TRAIN.ckpt_dir):
            os.mkdir(config.TRAIN.ckpt_dir)
        log_path = os.path.join(config.TRAIN.ckpt_dir, "log.csv")
        if not os.path.exists(log_path):
            with open(log_path, "w+") as out:
                writer = csv.writer(out)
                writer.writerow(["epoch", "dataset", "accuracy", "psnr_avg", "ssim_avg", "best", "best_sum"])
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tsrn', choices=['tsrn', 'tbsrn', 'tg', 'tpgsr', 'tatt'])
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='/root/data/TextZoom/test/easy')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--vis_dir', type=str, default=None)
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--patch_size', type=str, default="4,", help='1, 2, 4, 8, 16')
    parser.add_argument('--embed_dim', type=str, default="96,", help='')
    parser.add_argument('--window_size', type=str, default="2,", help='')
    parser.add_argument('--depths', type=str, default="1,", help='')
    parser.add_argument('--num_heads', type=str, default="6,", help='')
    parser.add_argument('--mlp_ratio', type=str, default="4,", help='')
    parser.add_argument('--drop_rate', type=str, default="0,", help='')
    parser.add_argument('--attn_drop_rate', type=str, default="0,", help='')
    parser.add_argument('--drop_path_rate', type=str, default="0.1,", help='')
    parser.add_argument('--rotate_train', type=float, default=0., help='')
    parser.add_argument('--rotate_test', type=float, default=0., help='')
    parser.add_argument('--stu_iter_b1', type=int, default=1, help='')
    parser.add_argument('--stu_iter_b2', type=int, default=1, help='')
    parser.add_argument('--tpg', default='visionlan', type=str, choices=['aster', 'moran', 'crnn', 'visionlan', None])
    parser.add_argument('--rec_path', type=str, default=None, help='')
    parser.add_argument('--font_path', type=str, default=None, help='')
    parser.add_argument('--sr_share', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.5, help='')
    parser.add_argument('--window_num', type=int, default=3, help='')
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    main(config, args)
