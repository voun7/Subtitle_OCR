import torch
import torch.nn as nn


class TextRecognitionModel(nn.Module):

    def __init__(self, num_classes: int, input_height: int, input_width: int, hidden_size: int = 256) -> None:
        """
        Convolutional Recurrent Neural Network (CRNN) for text recognition.
        :param num_classes: Number of output classes (e.g., alphabet characters)
        :param input_height: Height of the input image
        :param input_width: Width of the input image
        :param hidden_size: Size of the hidden LSTM layer. Default is 256.
        """
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        # Convolutional layers.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Recurrent layer (LSTM).
        self.rnn = nn.LSTM(128 * (input_width // 4), hidden_size, bidirectional=True)
        # Fully connected layer for classification.
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1, self.input_height, (self.input_width // 4))
        x = x.permute(3, 0, 1, 2)
        _, _, h, _ = x.size()
        x = x.contiguous().view(h, batch_size, -1)
        outputs, _ = self.rnn(x)
        x = self.fc(outputs[-1])
        return x
