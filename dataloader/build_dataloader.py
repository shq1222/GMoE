import torchvision
from torch.utils.data import DataLoader
import dataloader.datasets as datasets
import preprocess.transforms as transforms
from torch.utils.data import DataLoader


def generate_dataloader(batch_size, csv,data = "Daisee"):
    if data == "Daisee":

        dataset = datasets.Daisee(csv,
                            transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))
        return DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        # sampler=sampler
                        pin_memory=True,

                        )
                
    else:
        dataset = datasets.Emotion(csv,
                                transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))


    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4,
                      # sampler=sampler
                      pin_memory=True,
                    #   collate_fn=custom_collate_fn
                      )


def get_dataloader(batch_size, csv_train, csv_test,data="Daisee"):
    return {
        'train': generate_dataloader(batch_size, csv_train,data),
        'test': generate_dataloader(batch_size, csv_test,data)}

def get_D_E_dataloader(batch_size, csv_train, csv_test, data="Daisee"):
    return {
        'train': generate_dataloader(batch_size, csv_train,data),
        'test': generate_dataloader(batch_size, csv_test,"Emotion")}
