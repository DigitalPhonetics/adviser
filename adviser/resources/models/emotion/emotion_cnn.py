import torch.nn as nn
import torch


class cnn(nn.Module):
    def __init__(self, kernel_size, D_out, args):
        super(cnn, self).__init__()
        # Conv2d expects input as (N, Ch, H, W)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=args.kernels,
            kernel_size=kernel_size,
            stride=args.stride
        )
        self.dropout1 = nn.Dropout(p=args.dropout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=args.kernels,
            out_channels=args.kernels,
            kernel_size=args.height,
            stride=args.stride
        )
        self.dropout2 = nn.Dropout(p=args.dropout)
        self.relu2 = nn.ReLU()

        # output units of convolution ->(input_len-kernelsize)/stride + 1
        out_units_conv1 = (args.seq_length - args.height)/args.stride + 1
        out_units_conv2 = (out_units_conv1 - args.height)/args.stride + 1
        out_units = int(out_units_conv2) * args.kernels
        self.output = nn.Linear(out_units, D_out)

    def forward(self, x):
        # infer batch_size because it can be different in last batch
        batch_size = x.shape[1]
        # x shape: (seq(H), N, feat(W))
        # needs to be reshaped to (N, C, H, W)
        x_transformed = x.permute(1, 0, 2).unsqueeze(1)
        out = self.conv1(x_transformed)
        # shape after dropout: (batch_size, filter_maps, feats)
        out = self.dropout1(torch.squeeze(out, 3))
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        out = self.output(out.view(batch_size, -1))
        return out
