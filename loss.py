from torch import nn

class SegmantationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(SegmantationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
    def __call__(self, output, target, pixel_average=True):
        if pixel_average:
            return self.loss(output, target) #/ target.data.sum()
        else:
            return self.loss(output, target)