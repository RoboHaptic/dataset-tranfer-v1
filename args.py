import argparse


parser = argparse.ArgumentParser(description='Baseline method for 2-class MI task classification')

# Global
parser.add_argument('-dataset', type=str, default='iv_2a', help='dataset for train/test')
parser.add_argument('-load', type=str, default='', help='load model')
parser.add_argument('-gpu', type=int, default=0, help='Set gpu id')

# Training only
parser.add_argument('-train', action='store_true', help='A complete train')
parser.add_argument('-full_train', action='store_true', help='no validation and test')
parser.add_argument('-save', action='store_true', help='save model')
parser.add_argument('-batch_size', type=int, default=64, help='batch size')
parser.add_argument('-epoch', type=int, default=1000, help='total epochs')
parser.add_argument('-lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('-n_workers', type=int, default=1, help='num_workers')
parser.add_argument('-set_weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('-lambd', type=float, default=0.5, help='weight for balancing loss function')
parser.add_argument('-step', action='store_false', help='To cancel scheduler')


# Test setting
parser.add_argument('-test', action='store_true', help='Only test')
parser.add_argument('-visualizer', action='store_true', help='Visualization')

args = parser.parse_args()
