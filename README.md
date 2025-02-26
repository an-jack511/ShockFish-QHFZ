1. input
2. [conv2D: 256 * (3*3filter, stride = 1)] + BN + relu
3. [ResNet: conv2D + BN + relu + conv2D + BN && skip connect + relu] * 19 + relu
4. + PolicyHead: [conv2D: 2 * (1*1filter, stride = 1)] + BN + tanh
   + ValueHead: [conv2D: 2 * (1*1filter, stride = 1)] + BN + softmax，因为用了 softmax，所以在 MCTS 时要复原。

DNN：$(\bold{p}, v) = f_\theta(s)$
loss func : $$l = (z - v)^2 - \boldsymbol{\pi}^\top\log\bold{p} + c||\theta||^2$$

$z$：实际 MCTS 获胜方
$v$：DNN 预测获胜概率
$\boldsymbol{\pi}$：DNN 走法策略预测
$\bold{p}$：实际 MCTS 走法策略预测（可能走法平均价值向量）
$c, \theta$：防过拟合