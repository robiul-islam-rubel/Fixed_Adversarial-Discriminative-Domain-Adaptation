import torch.nn.functional as F
from torch import nn


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, input_channels=1):
        """Init LeNet encoder.
        Args:
            input_channels (int): number of input channels (1 for grayscale, 3 for RGB)
        """
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(input_channels, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self, num_classes=10):
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, feat):
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
