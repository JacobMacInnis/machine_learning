import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate data: y = sin(x)
x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100).unsqueeze(1)
y = torch.sin(x)

# Define neural network with 1 hidden layer
class SineApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SineApproximator()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
losses = []
for epoch in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# Predict after training
with torch.no_grad():
    predicted = model(x)

# Plot the true vs predicted function
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x.numpy(), y.numpy(), label='True y = sin(x)', color='blue')
plt.plot(x.numpy(), predicted.numpy(), label='Model Prediction', color='red')
plt.title("Function Approximation")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.tight_layout()
plt.show()
