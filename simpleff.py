import torch

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# load dataset

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        ).to(device)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)
  print(t, loss.item())
  
  model.zero_grad()

  loss.backward()

  with torch.no_grad():
    for param in model.parameters():
      param.data -= learning_rate * param.grad
