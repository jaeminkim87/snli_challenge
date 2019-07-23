import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Callable, List


class Preprocessing():
    """preprocessing"""

    def __init__(self, args) -> None:
        file_path = args.file_path
        self.data_path = args.data_path
        self.train_path = args.file_path + '/snli_1.0_train.txt'
        self.test_path = args.file_path + '/snli_1.0_test.txt'
        self.dev_path = args.file_path + '/snli_1.0_dev.txt'

    def get_data(self):
        traing_data = self.train_path
        test_data = self.test_path
        dev_data = self.dev_path

        return traing_data, test_data, dev_data

    def make_datafile(self):
        train_data, test_data, dev_data = self.get_data()
        data = pd.read_csv(train_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
        data = data[data.gold_label != '-']
        data.replace('N/A', np.nan, inplace=True)
        data.sentence1.replace("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=", "", regex=True, inplace=True)
        data.sentence2.replace("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=", "", regex=True, inplace=True)
        data.replace('', np.nan, inplace=True)
        data.dropna(axis=0, how='any')
        train_split, val_split = train_test_split(data, test_size=0.2, shuffle=True, random_state=777)
        train_split.to_csv(self.data_path + '/train.txt', sep='\t', index=False)
        val_split.to_csv(self.data_path + '/val.txt', sep='\t', index=False)
        print("Complete make train, val txt file.")

        data = pd.read_csv(dev_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
        data = data[data.gold_label != '-']
        data.replace('N/A', np.nan, inplace=True)
        data.sentence1.replace("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=", "", regex=True, inplace=True)
        data.sentence2.replace("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=", "", regex=True, inplace=True)
        data.replace('', np.nan, inplace=True)
        data.dropna(axis=0)
        data.to_csv(self.data_path + '/dev.txt', sep='\t', index=False)

        print("Complete make dev txt file.")

        data = pd.read_csv(test_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
        data = data[data.gold_label != '-']
        data.replace('N/A', np.nan, inplace=True)
        data.sentence1.replace("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=", "", regex=True, inplace=True)
        data.sentence2.replace("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=", "", regex=True, inplace=True)
        data.replace('', np.nan, inplace=True)
        data.dropna(axis=0)
        data.to_csv(self.data_path + '/test.txt', sep='\t', index=False)
        print("Complete make test txt file.")
