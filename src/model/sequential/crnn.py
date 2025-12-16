import torch.nn as nn

TARGET_SIZE = (24, 24)

class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=128, cnn_out_channels=256, dropout=0.35, num_rnn_layers=1):
        super(CRNN, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_out_channels = cnn_out_channels
        self.dropout = dropout

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  

            nn.Conv2d(128, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
        )

        self.cnn_dropout = nn.Dropout2d(dropout)

        rnn_input_size = cnn_out_channels * 16

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,  
            bidirectional=True,
            batch_first=False,  
            dropout=dropout if num_rnn_layers > 1 else 0  
        )

        self.rnn_dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)

        conv_out = self.cnn_dropout(conv_out)

        B, _, _, W = conv_out.size()

        conv_out = conv_out.permute(0, 3, 1, 2)
        conv_out = conv_out.contiguous().view(B, W, -1) 

        rnn_input = conv_out.permute(1, 0, 2)

        rnn_output, _ = self.rnn(rnn_input)

        rnn_output = self.rnn_dropout(rnn_output)

        output = self.fc(rnn_output)

        return output