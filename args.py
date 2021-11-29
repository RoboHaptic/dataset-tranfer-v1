import argparse


parser = argparse.ArgumentParser(description='Subject-independent classification with KU Data')

# Global
parser.add_argument('-dataset', type=str, default='iv_2a', help='dataset for train/test')
parser.add_argument('-load', type=str, default='', help='load model')

# Training only
parser.add_argument('-train', action='store_true', help='A complete train')
parser.add_argument('-full_train', action='store_true', help='no validation and test')
parser.add_argument('-save', action='store_true', help='save model')
parser.add_argument('-gpu', type=int, default=0, help='set which gpu to use')
parser.add_argument('-batch_size', type=int, default=64, help='batch size')
parser.add_argument('-epoch', type=int, default=1200, help='total epochs')
parser.add_argument('-lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('-set_weight_decay', type=float, default=1, help='weight_decay')

# Test setting
parser.add_argument('-test', action='store_true', help='Only test')
parser.add_argument('-visualizer', action='store_true', help='Visualization')

args = parser.parse_args()
