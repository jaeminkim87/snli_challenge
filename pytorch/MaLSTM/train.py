import argparse
import torch


def train(args, data):
 return ""

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-path', default='./data_in')
	parser.add_argument('--epoch', default=1, type=int)
	parser.add_argument('--batch-size', default=32, type=int)
	parser.add_argument('--data-type', default='SNLI')
	parser.add_argument('--dropout', default=0.25, type=float)
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--learning-rate', default=0.5, type=float)
	parser.add_argument('--weight-decay', default=5e-5, type=float)
	parser.add_argument('--word-dim', default=300, type=int)
	parser.add_argument('--classes', default=2, type=int)

	args = parser.parse_args()




if __name__ == '__main__':
	main()
