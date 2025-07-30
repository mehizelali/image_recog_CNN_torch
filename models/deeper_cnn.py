import torch 
import torch.nn as nn 


class DeepCNN(nn.Module):
    def __init__(
            self, 
            img_size, 
            in_chans, 
            out_chans1, 
            out_chans2, 
            kernal_size, 
            pool_size, 
            hidden_units, 
            num_classes, 
            dropout
            ):
        super(DeepCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans1, kernel_size=kernal_size),  # Conv1
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chans1, out_channels=out_chans2, kernel_size=kernal_size),  # Conv2
            nn.BatchNorm2d(out_chans2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),  # Pool1

            nn.Conv2d(in_channels=out_chans2, out_channels=out_chans2, kernel_size=kernal_size),  # Conv3
            nn.ReLU(), 
            nn.Conv2d(in_channels=out_chans2, out_channels=out_chans2, kernel_size=kernal_size),  # Conv4
            nn.BatchNorm2d(out_chans2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=pool_size)  # Pool2
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_chans, img_size, img_size)
            dummy_output = self.cnn(dummy_input)
            flattened_dim = dummy_output.view(1, -1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(in_features=flattened_dim, out_features=hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_units, out_features=num_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x: torch.tensor):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x 
