from torch.utils.data import DataLoader
def createDataLoader(data_name='Zhoukun', shuffle=True, split='frontal',  batch_size=1, num_workers=5):
    if split == 'uv':
        from .dataset_uv import DataUniversal
        return DataLoader(dataset=DataUniversal(name=data_name), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    if split == 'frontal':
        from .dataset_frontal import DataUniversal
        return DataLoader(dataset=DataUniversal(name=data_name), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)