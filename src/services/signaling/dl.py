class SignalingDataLoader:

    def __init__(self, config):
        self.config = config

    def fetch_data(self):
        """Fetch raw data"""
        pass

    def fetch_training_dataloader(self):
        """Return training dataloader"""
        pass

    def fetch_test_dataset(self):
        """Return test dataset"""
        pass
