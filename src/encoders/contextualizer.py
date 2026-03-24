from text_handler import FlangService





class Contextualizer:

    def __init__(self,config):
        self.config = config
        self.d_model = self.config.d_model
    
    def contextualize(self, equity, timestamps,ts,text):
        
        torch.randn(ts.shape[0], self.d_model)
        return 