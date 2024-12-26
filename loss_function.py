"""
loss function

build the cls. + hmo. loss function
"""
from resource import *


class HomologyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, feature: Dict[str, Tensor]) -> Tensor:
        key, feat, n = feature.keys(), list(feature.values()), len(feature)
        loss = [self.loss(feat[idx], feat[(idx+1) % n]) for idx in range(n)]
        return sum(loss)


class CombinedLoss(nn.Module):
    def __init__(self, ratio: float = loss_ratio):
        super().__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.homology_loss = HomologyLoss()
        self.ratio = ratio
        if loss_ratio < 0 or loss_ratio > 1.0:
            raise ValueError('Invalid Loss Ratio: out of range [0, 1]')
        print(f"loss ratio = {self.ratio:.2f} : {1-self.ratio:.2f}")

    def forward(self,
                output: Tensor,
                feature: Dict[str, Tensor],
                label: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cls_loss = self.classify_loss(output, label)
        hmo_loss = self.homology_loss(feature)
        cb_loss = cls_loss*self.ratio+hmo_loss*(1.0-self.ratio)
        return cb_loss, cls_loss, hmo_loss

    def update_ratio(self, print_log: bool = True):
        self.ratio = round(max(self.ratio-delta_ratio, min_ratio), 4)
        if print_log:
            print(f"loss ratio -> {self.ratio:.2f} : {1-self.ratio:.2f}")


if __name__ == "__main__":
    criterion = CombinedLoss()
    print(criterion)
