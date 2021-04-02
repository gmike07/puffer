op_version_set = 0
def forward(self,
    input_1: Tensor) -> Tensor:
  input_2 = torch.addmm(getattr(self, "0").bias, input_1, torch.t(getattr(self, "0").weight), beta=1, alpha=1)
  input_3 = torch.threshold(input_2, 0., 0.)
  input_4 = torch.addmm(getattr(self, "2").bias, input_3, torch.t(getattr(self, "2").weight), beta=1, alpha=1)
  input = torch.threshold(input_4, 0., 0.)
  return input
