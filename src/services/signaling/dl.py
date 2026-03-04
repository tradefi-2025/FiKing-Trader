import torch
class SignalingDataLoader:

    def __init__(self, metadata,config):
        self.config = config

    def fetch_data(self):
        """Fetch raw data"""
        pass

    def fetch_training_dataloader(self):
        dummy_data = torch.randn(100, self.config.d_model), torch.randn(100, 1)
        dummy_test_data = torch.randn(20, self.config.d_model), torch.randn(20, 1)
        dataset = torch.utils.data.TensorDataset(*dummy_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        return dataloader,dummy_test_data
        """Return training dataloader"""
        pass

    def fetch_test_dataset(self):
        """Return test dataset"""
        pass
