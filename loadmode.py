import torch

checkpoint = torch.load('./check_point/jdata2024-06-14 20_19_26.pth')
print(checkpoint.keys())
print(checkpoint['Graph_encoder.view.gnn_layers.0.bias'])