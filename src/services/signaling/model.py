import os
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .config import SignalingConfig
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

class SignalingModelV1(nn.Module):

    def __init__(self, config):
        super(SignalingModelV1,self).__init__()
        self.config = config
        d_model=self.config.d_model
        self.layer_norm1 = nn.LayerNorm(d_model//2)  # For o1+o3 
        self.layer_norm2 = nn.LayerNorm(d_model//4)  # For o5+o2
        self.layer1=nn.Linear(d_model,d_model//2)
        self.layer2=nn.Linear(d_model//2,d_model//4)
        self.layer3=nn.Linear(d_model//4,d_model//2)
        self.layer4=nn.Linear(d_model//2,d_model//8)
        self.layer5=nn.Linear(d_model//8,d_model//4)
        self.out = nn.Linear(d_model//4,2)


        
    def forward(self,x):
        B,D=x.shape
        activation = self.config.activation
        o1 = activation(self.layer1(x))
        o2 = activation(self.layer2(o1))
        o3 = self.layer3(o2)
        o3 = self.layer_norm1(o1+o3)  # Use layer_norm1 for d_model//2
        o4 = activation(self.layer4(o3))
        o5 = self.layer5(o4)
        o5 = self.layer_norm2(o5+o2)  # Use layer_norm2 for d_model//4
        out = self.out(o5)

        mu= out[:,0].unsqueeze(-1)
        logvar=out[:,1].unsqueeze(-1)
        
        pred= logvar.exp()*torch.randn((B,1),requires_grad=False) 
        return pred, mu, logvar

    def reconstruction_loss(self,pred,Y):
        return F.mse_loss(pred,Y)
    
    def fit(self, dataloader):
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config.lr) 
        
        for epoch in range(self.config.epoch):
            l=0
            for X, Y in dataloader:
                optimizer.zero_grad()
                pred, _,_ = self(X)
                loss=self.reconstruction_loss(pred,Y)
                l+=loss.item()
                loss.backward()
                optimizer.step()

            logger.info(f"loss epoch{epoch} / {self.config.epoch} : {l/len(dataloader)}")

    def test(self,test_dataset):
        self.eval()
        with torch.no_grad():
            X, Y = test_dataset.tensors
            pred, _,_ = self(X)
            loss=self.reconstruction_loss(pred,Y)
        logger.info(f"Test loss: {loss.item()}")
        plt.plot(pred.cpu().numpy(), label="Predicted")
        plt.plot(Y.cpu().numpy(), label="Actual")
        plt.legend()
        plt.title("Predicted vs Actual")
        self._finalize_plot("pred_vs_actual")
        

    def inference(self, input_data, confidence_level):
        with torch.no_grad():
            pred, mu, logvar = self(input_data)
            dist = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
            res = {
                "estimated price"  : round(pred.item(),2),
                "prob"             : round(dist.log_prob(pred).exp().item(),2),  # PDF at pred
                "mu"               : round(mu.item(),2),
                "logvar"           : round(logvar.item(),2),
                "estimated action" : t_test(mu, logvar, confidence_level).item()
            }
        return res
    

        
def t_test(mu: torch.Tensor, logvar: torch.Tensor, confidence_level: float) -> torch.Tensor:
    """
    Performs t test of a gaussian distribution and the value 0.
    If the distribution is confidently positive, returns 1.
    If it is confidently negative, returns -1. Otherwise, zero.
    """
    std = torch.exp(0.5 * logvar)                          # σ from log-variance

    alpha = 1.0 - confidence_level
    # z critical value via torch — equivalent to scipy.stats.norm.ppf(1 - alpha/2)
    standard_normal = Normal(
        torch.zeros_like(mu),
        torch.ones_like(mu)
    )
    z = standard_normal.icdf(torch.tensor(1 - alpha / 2, device=mu.device, dtype=mu.dtype))

    lower = mu - z * std
    upper = mu + z * std

    result = torch.where(lower > 0,  torch.ones_like(mu),
             torch.where(upper < 0, -torch.ones_like(mu),
                                     torch.zeros_like(mu)))
    return result


def test_signaling_model():

    config = SignalingConfig()
    config.epoch = 1
    model = SignalingModelV1(config)

    # Dummy data
    X = torch.randn(100, config.d_model)
    Y = torch.randn(100, 1)

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model.fit(dataloader)

    input_data = torch.randn(1, config.d_model)
    confidence_level = 0.95
    result = model.inference(input_data, confidence_level)
    logger.info("Inference result: %s", result)

if __name__ == "__main__":
    test_signaling_model()


class SignalingModelV2(nn.Module):

    def __init__(self, config):
        super(SignalingModelV2, self).__init__()
        self.config = config
        self.sequential = nn.Sequential(
            nn.Linear(config.d_model, 1)
        )
    
    
    def reconstruction_loss(self,pred,Y):
        return F.mse_loss(pred,Y)
    

    def fit(self, dataloader):
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config.lr) 
        
        for epoch in range(self.config.epoch):
            l=0
            for X, Y in dataloader:
                optimizer.zero_grad()
                pred= self(X)
                loss=self.reconstruction_loss(pred,Y)
                l+=loss.item()
                loss.backward()
                optimizer.step()

            logger.info(f"loss epoch{epoch} / {self.config.epoch} : {l/len(dataloader)}")

    def forward(self, input_data):
        return self.sequential(input_data)
    
    def test(self,test_dataset):
        self.eval()
        with torch.no_grad():
            X, Y = test_dataset.tensors
            pred= self(X)
            loss=self.reconstruction_loss(pred,Y)

        logger.info(f"Test loss: {loss.item()}")
        # still on CPU here (use .cpu() earlier if needed)
        pred_cpu = pred.detach().cpu()
        y_cpu = Y.detach().cpu()

        idx = torch.arange(pred_cpu.shape[0])

        plt.scatter(idx.numpy(), pred_cpu.numpy(), label="Predicted")
        plt.scatter(idx.numpy(), y_cpu.numpy(), label="Actual")
        plt.legend()
        plt.title("Predicted vs Actual over index")
        self._finalize_plot("pred_vs_actual_index")

    def _finalize_plot(self, name):
        if threading.current_thread() is threading.main_thread():
            plt.show()
            return

        plot_dir = os.getenv("SIGNALING_PLOT_DIR", "logs/signaling_plots")
        os.makedirs(plot_dir, exist_ok=True)
        filename = os.path.join(plot_dir, f"{name}_{os.getpid()}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

    def inference(self, input_data, confidence_level):
        with torch.no_grad():
            pred = self(input_data)
            
            res = {
                "estimated price"  : round(pred.item(),2),
                "estimated action" : t_test(pred, torch.ones_like(pred), confidence_level).item()
            }
        return res