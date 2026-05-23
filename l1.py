
  l1_lambda = 1e-2 * (1 - 0.9 * epoch / self.epochs)
  for k, m in self.model.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        m.weight.grad.data.add_(l1_lambda * torch.sign(m.weight.data))
       m.bias.grad.data.add_(1e-2 * torch.sign(m.bias.data))
