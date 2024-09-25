import torch.nn as nn

from .rec_ctc_head import CTCHead
from ..necks.rnn import Im2Seq, SequenceEncoder


class MultiHead(nn.Module):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop('head_list')
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == 'SARHead':
                pass
            elif name == 'NRTRHead':
                pass
            elif name == 'CTCHead':
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]['Neck']
                encoder_type = neck_args.pop('name')
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels, encoder_type=encoder_type, **neck_args)
                # ctc head
                self.ctc_head = CTCHead(in_channels=self.ctc_encoder.out_channels,
                                        out_channels=out_channels_list['CTCLabelDecode'])
            else:
                raise NotImplementedError('{} is not supported in MultiHead yet'.format(name))

    def forward(self, x):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder)
        return ctc_out
