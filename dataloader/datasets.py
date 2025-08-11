from torch.utils.data import Dataset
import pandas as pd
import os

class Daisee(Dataset):
    def __init__(self, csv, transform=None, target_count=1696):
        self.dataframe = pd.read_csv(csv)
        self.transform = transform


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # 选择 Engagement 作为目标
        Engagement = self.dataframe.iloc[index].label2
        video = self.dataframe.iloc[index].path
        video = self.transform(video)  # type: ignore
        return video, Engagement
    
    
class Emotion(Dataset):

    def __init__(self, csv, transform=None):

        self.dataframe = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        label = self.dataframe.iloc[index].label
        video_path = self.dataframe.iloc[index].path
        vid = os.path.basename(video_path)
        video_trans = self.transform(video_path) if self.transform else video_path

        return video_trans, label
