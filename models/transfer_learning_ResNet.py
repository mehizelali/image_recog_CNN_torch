from torchvision.models import resnet50
import torch.nn as nn 


def ResNet50_pretrained(hidden_features, num_classes, dropout):
    model = resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=hidden_features),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_features=hidden_features, out_features=num_classes)
)


    return model 