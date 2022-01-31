# torch
import torch
from torch.utils.data import Dataset

class FeedbackDataset(Dataset):
    """
    Class for transforming feedback corpuses of text into labels
    """
    
    def __init__(self, df, text_list, type_labels, file_mode=True, word_pp = None):
        """
        df: Pandas DataFrame containing proper label data
        text_list: Either a list of paths to text or a list of tuples with ids and text
        type_labels: Dictionary which transforms discourse type into integers. Must contain a 'None' entry
        file_mode: Boolean indicating whether you want text list to contain paths or actual strings of text
        word_pp: Processing to do on words.
        """
        self.df = df
        self.groups = df.groupby('id').groups
        self.text_list = text_list
        self.type_labels = type_labels
        self.file_mode = file_mode
        self.word_pp = word_pp

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        if self.file_mode:
            id = self.text_list[idx].split('/')[-1].replace('.txt', '')
            data = open(self.text_list[idx]).read()
        else:
            id = self.text_list[idx][0]
            data = self.text_list[idx][1]
        data = data.split()

        # Get word-level labels
        doc = self.df.iloc[self.groups[id]]
        labels = torch.full((len(data),), self.type_labels['None'])
        for distype, preds in zip(doc['discourse_type'], doc['predictionstring']):
            indices = [int(i) for i in preds.split()]
            labels[indices] = self.type_labels[distype]

        # Preprocess words
        if self.word_pp:
            data = self.word_pp(data)
            
        return data, labels
