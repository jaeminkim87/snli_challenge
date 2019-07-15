import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Callable, List


class preprocessing():
    """preprocessing"""

    def __init__(self, data_path, options) -> None:
        file_path = options.filepath
        self.data_path = data_path
        self.train = file_path.train
        self.test = file_path.test
        self.dev = file_path.dev

    def getData(self):
        traing_data = self.data_path + '/' + self.train
        test_data = self.data_path + '/' + self.test
        dev_data = self.data_path + '/' + self.dev

        return traing_data, test_data, dev_data

    def makeDataFile(self):
        train_data, test_data, dev_data = self.getData()
        data = pd.read_csv(train_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
        data = data[data.gold_label != '-']
        train_split, val_split = train_test_split(data, test_size=0.2, shuffle=True, random_state=777)
        train_split.to_csv(self.data_path + '/train.txt', sep='\t', index=False, header=False)
        val_split.to_csv(self.data_path + '/val.txt', sep='\t', index=False, header=False)
        print("Complete make train, val txt file.")

        data = pd.read_csv(dev_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
        data = data[data.gold_label != '-']
        data.to_csv(self.data_path + '/dev.txt', sep='\t', index=False, header=False)

        print("Complete make dev txt file.")

        data = pd.read_csv(test_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
        data = data[data.gold_label != '-']
        data.to_csv(self.data_path + '/test.txt', sep='\t', index=False, header=False)
        print("Complete make test txt file.")
