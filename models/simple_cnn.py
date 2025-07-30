import torch 
import torch.nn as nn


class SimpleCNN(nn.Module):
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
            dropout: float = 0.1
            ):
        super(SimpleCNN, self).__init__()

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
        self.mlp = nn.Sequential(
            nn.Linear(in_features=flattened_dim, out_features=hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=num_classes)
        )



    def forward(self, x: torch.tensor):
        
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x 