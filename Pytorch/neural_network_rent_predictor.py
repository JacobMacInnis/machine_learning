import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------
# 1. Simulated Housing Dataset
# ------------------------
# Features: [sqft, bedrooms, distance_to_city (mi), has_parking, year_built]
# Target: monthly rent ($)
x = torch.tensor([
    [850, 2, 5.0, 1, 2005],
    [1200, 3, 3.0, 1, 2015],
    [650, 1, 10.0, 0, 1990],
    [1500, 4, 2.0, 1, 2020],
    [400, 1, 12.0, 0, 1980],
    [1000, 3, 4.5, 1, 2000],
    [900, 2, 6.0, 0, 1995],
    [1100, 3, 3.5, 1, 2010],
    [700, 1, 11.0, 0, 1985],
    [1300, 4, 2.5, 1, 2018],
], dtype=torch.float32)

y = torch.tensor([
    [1800],
    [2500],
    [1200],
    [3000],
    [900],
    [2100],
    [1600],
    [2300],
    [1100],
    [2700],
], dtype=torch.float32)

# Normalize features
x_mean = x.mean(0, keepdim=True)
x_std = x.std(0, keepdim=True)
x_norm = (x - x_mean) / x_std

# ------------------------
# 2. Define the Model
# ------------------------
class RentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = RentPredictor()

# ------------------------
# 3. Train the Model
# ------------------------
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(500):
    y_pred = model(x_norm)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# ------------------------
# 4. Plot Loss
# ------------------------
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# ------------------------
# 5. Predict a New Example
# ------------------------
# Predict rent for: 1000 sqft, 3 beds, 4 miles out, has parking, built 2012
new_house = torch.tensor([[1000, 3, 4.0, 1, 2012]], dtype=torch.float32)
new_house_norm = (new_house - x_mean) / x_std

with torch.no_grad():
    prediction = model(new_house_norm)
    print(f"Predicted monthly rent: ${prediction.item():.2f}")
