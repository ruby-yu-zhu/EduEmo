from torch.utils.data import Dataset
import pandas as pd
import torch

class LoadDataClass(Dataset):
    def __init__(self, args, filename, savepath):
        self.args = args
        self.filename = filename
        self.savepath = savepath
        self.data, self.labels = self.load_dataset()
        self.inputs, self.lengths, self.label_indices, self.edu_token_lists = self.load_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        df = pd.read_csv(self.filename, sep='\t')
        x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values
        return x_train, y_train

    def load_lists(self,file):
        with open(file) as f:
            lists = []
            for line in f:
                line = line.strip().split(' ')
                line = list(map(int, line))
                lists.append(line)
           
        return lists
    


    def load_list(self,file):
        with open(file) as f:
            for line in f:
                line =line.strip().split(' ')
            lists = list(map(int, line))
        return lists
    
    def load_data(self):
        inputs = self.load_lists(self.savepath+'inputs.txt')
        lengths = self.load_list(self.savepath+'data_length.txt')
        label_indices = self.load_lists(self.savepath+'label_indices.txt')
        edu_token_lists = self.load_lists(self.savepath+'edu_token_lists.txt')

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        #edu_token_lists = torch.tensor(edu_token_lists, dtype=torch.long)
        
        return inputs, data_length, label_indices, edu_token_lists

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        edu_token_list = self.edu_token_lists[index]
        return inputs, labels, length, label_idxs, edu_token_list

    def __len__(self):
        return len(self.inputs)
