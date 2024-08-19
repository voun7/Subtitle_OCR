import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels=6625, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels, bias=True)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels, bias=True)
            self.fc2 = nn.Linear(mid_channels, out_channels, bias=True)

        self.out_channels = out_channels
        self.mid_channels = mid_channels

    def forward(self, x):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        predicts = predicts.permute(1, 0, 2)  # B, T, C --> T, B, C  (Input sequence length, Batch size, No of classes)
        predicts = predicts.log_softmax(2)
        return predicts
