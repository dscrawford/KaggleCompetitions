# native
import os

# data
import pandas as pd

def identity_fun(x):
    return x

class FeedbackPrize:
    """
    Class for reading feedback kaggle competition data
    """
    
    def __init__(self, path):
        """
        path: str representing path to feedback data
        """
        self.path = path
        self.csv_path = os.path.join(path, 'train.csv')
        self.sample_path = os.path.join(path, 'sample_submission.csv')
        self.text_path = os.path.join(path, 'train')
        self.text_paths = [os.path.join(self.text_path, text) for text in os.listdir(self.text_path)]
        
    def get_df(self):
        return pd.read_csv(self.csv_path)
        
    def get_sample_submission(self):
        return pd.read_csv(self.sample_path)

    def get_all_text(self):
        return [(path.split('/')[-1].replace('.txt', ''), open(path).read()) for path in self.text_paths]

    def get_text(self, i):
        return self.text_paths[i].split('/')[-1].replace('.txt', ''), open(self.text_paths[i]).read()
