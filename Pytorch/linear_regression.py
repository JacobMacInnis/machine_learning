import torch
import torch.nn as nn
import torch.optim as optim

# Training data (x, y) where y = 2x + 1
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# Define a super simple linear model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)  # y = wx + b

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# Loss function: Mean Squared Error
loss_fn = nn.MSELoss()

# Optimizer: Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    y_pred = model(x)                      # Forward pass
    loss = loss_fn(y_pred, y)              # Compute loss

    optimizer.zero_grad()                  # Reset gradients to zero
    loss.backward()                        # Backward pass (compute gradients)
    optimizer.step()                       # Update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Try predicting a new value
with torch.no_grad():
    test = torch.tensor([[5.0]])
    prediction = model(test)
    print("Prediction for x=5:", prediction.item())  # Should be close to 11
