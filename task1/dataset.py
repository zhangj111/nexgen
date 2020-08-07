from torch.utils.data import Dataset
import torch
import os


class EHDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # Load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['lines_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_line'][i]), \
               torch.FloatTensor([self.data['labels'][i]])

    def __len__(self):
        return len(self.data['labels'])