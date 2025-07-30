import torch 
import torch.nn as nn 
from .Fast_RBF_KAN import RBFKAN



class ConvRBFKAN(nn.Module):
    def __init__(
            self,
            img_size: int = 64,
            in_chans: int = 3,  
            out_chans1: int = 32, 
            out_chans2: int = 64,
            kernal_size : int= 3,
            pool_size: int = 2,
            hidden_dim: int = 64,
            num_classes: int = 10,
            num_grids:int = 1,
            dropout: float = 0.1
            ):
        super(ConvRBFKAN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans1,kernel_size=kernal_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
            
            nn.Conv2d(in_channels=out_chans1, out_channels=out_chans2, kernel_size=kernal_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
            
        )
        # Dynamically compute output size after CNN
        def compute_output_dim(size, kernel, pool):
            size = (size - kernel + 1)  # Conv1
            size = (size - pool) // pool + 1  # Pool1
            size = (size - kernel + 1)  # Conv2
            size = (size - pool) // pool + 1  # Pool2
            return size

        conv_output_size = compute_output_dim(img_size, kernal_size, pool_size)
        flattened_dim = out_chans2 * conv_output_size * conv_output_size
        self.rbfkan = RBFKAN(layers_hidden=[flattened_dim, hidden_dim, num_classes],num_grids=num_grids)


    def forward(self, x: torch.tensor):

        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.rbfkan(x)

        return x