import itertools
import pickle
import pandas as pd
import gluonnlp as nlp

class build_vocab():
	def __init__(self, args):
		super(build_vocab, self).__init__()
		
		file_path = args['file_path']
		self.train_path = file_path + '/' + args['train']
		
	def make_vacabfile():
		tr = pd.read_csv(self.train_path, sep='\t'
