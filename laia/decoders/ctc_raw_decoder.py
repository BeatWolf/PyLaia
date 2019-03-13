from __future__ import absolute_import

#import pywrapfst as fst
from torch.nn.functional import log_softmax

from laia.losses.ctc_loss import transform_output

import laia.common.logging as log

class CTCRawDecoder(object):
    def __init__(self, normalize=False):
        self._normalize = normalize

    def __call__(self, x):
        x, xs = transform_output(x)
        # Normalize log-posterior matrices, if necessary
        if self._normalize:
            x = log_softmax(x, dim=2)
        x = x.permute(1, 0, 2).cpu()
        self._output = []
        D = x.size(2)
        for logpost, length in zip(x, xs):
            temp = []
            for t in range(length):
                for j in range(D):
                    #log.error(str([t, j, float(-logpost[t, j])]))
                    temp.append([t, j, float(logpost[t, j])])
            self._output.append(temp)
            
        return self._output

    @property
    def output(self):
        return self._output
