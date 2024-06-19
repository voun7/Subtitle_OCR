# Convolutional Recurrent Neural Network algorithm  https://arxiv.org/abs/1507.05717
import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_):
        recurrent, _ = self.rnn(input_)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, output_size]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, image_height: int, num_class: int, channel_size: int = 3, hidden_size: int = 256,
                 leaky_relu: bool = False) -> None:
        """
        :param image_height: the height of the input image to network
        :param channel_size: the channel size of the input image
        :param num_class: the number of characters to recognize e.g. 26 for english alphabets
        :param hidden_size: size of the lstm hidden state
        :param leaky_relu:
        """
        super().__init__()
        assert image_height % 16 == 0, 'image height has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]  # kernel_sizes
        ps = [1, 1, 1, 1, 1, 1, 0]  # paddings
        ss = [1, 1, 1, 1, 1, 1, 1]  # strides
        nm = [64, 128, 256, 256, 512, 512, 512]  # channels

        cnn = nn.Sequential()

        def conv_relu(i, batch_normalization=False):
            input_channel = channel_size if i == 0 else nm[i - 1]
            output_channel = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(input_channel, output_channel, ks[i], ss[i], ps[i]))
            if batch_normalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(output_channel))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(BidirectionalLSTM(512, hidden_size, hidden_size),
                                 BidirectionalLSTM(hidden_size, hidden_size, num_class))

    def forward(self, input_):
        # conv features
        conv = self.cnn(input_)
        b, c, h, w = conv.shape
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # [b, c, 1, w] -> [b, c, w]
        conv = conv.permute(2, 0, 1)  # [b, c, w] -> [w, b, c]
        # rnn features
        output = self.rnn(conv)
        # add log_softmax to converge output
        output = output.log_softmax(2)
        return output


if __name__ == '__main__':
    test_img = torch.rand([4, 3, 32, 320])  # Batch Size, Image Channel, Image Height, Image Width
    test_model = CRNN(**{"image_height": 32, "num_class": 50})
    test_output = test_model(test_img)
    print(test_model), print(test_output), print(test_output.shape)
