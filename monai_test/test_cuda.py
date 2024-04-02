import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ",device)
print("torch version : ",torch.__version__)
# model = model.to(device)
