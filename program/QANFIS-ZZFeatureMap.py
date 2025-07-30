# --- 1. Setup dan Seed ---
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Sampler
from qiskit.circuit import QuantumCircuit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Load & Normalize Dataset ---
df = pd.read_csv("dummy_flood_data4.csv")
X = df[['rainfall', 'water_level']].values
y = df['flood_potential'].values.reshape(-1, 1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.25, random_state=SEED)

# --- 3. Trainable Gaussian Membership Function ---
class TrainableGaussianMF(nn.Module):
    def __init__(self, num_mf):
        super().__init__()
        self.mean = nn.Parameter(torch.linspace(0.3, 0.7, num_mf))
        self.sigma = nn.Parameter(torch.full((num_mf,), 0.15))

    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

# --- 4. Quantum Circuit Setup (4 Qubits) ---
num_rules = 4  # 2 MF × 2 MF
feature_map = ZZFeatureMap(feature_dimension=num_rules, reps=1)
ansatz = RealAmplitudes(num_qubits=num_rules, reps=1, entanglement='linear')

qc = QuantumCircuit(num_rules)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

qnn = SamplerQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    sampler=Sampler()
)
torch_qnn = TorchConnector(qnn).to(device)

# --- 5. QANFIS Model ---
class QANFISNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_mf = 2
        self.num_rules = self.num_mf ** 2
        self.mf_rain = TrainableGaussianMF(self.num_mf)
        self.mf_level = TrainableGaussianMF(self.num_mf)
        self.qnn = torch_qnn

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        mf1 = self.mf_rain(x1)
        mf2 = self.mf_level(x2)
        rule_activation = torch.einsum("bi,bj->bij", mf1, mf2).reshape(-1, self.num_rules)
        weights = rule_activation / rule_activation.sum(dim=1, keepdim=True)

        outputs = torch.stack([self.qnn(weights[i]) for i in range(weights.size(0))])
        bit_weights = torch.arange(outputs.shape[1], dtype=torch.float32, device=outputs.device)
        expectation = torch.sum(outputs * bit_weights, dim=1) / (outputs.shape[1] - 1)

        return expectation

# --- 6. Training ---
model = QANFISNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

losses = []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f}")

# --- 7. Evaluation ---
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_test = model(X_test_tensor).cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

# --- 8. Create Output Directory ---
os.makedirs("output", exist_ok=True)

# --- 9. Visualization ---

# Scatter: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color="green")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Flood Potential")
plt.ylabel("Predicted (QANFIS)")
plt.title(f"QANFIS (4 Qubit) - RMSE: {rmse:.4f}, R²: {r2:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/qanfis_scatter_4qubit_data4.png", dpi=300)
plt.show()

# Training Loss
plt.figure(figsize=(8, 4))
plt.plot(losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QANFIS Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output/qanfis_loss_4qubit_data4.png", dpi=300)
plt.show()

# Output vs Actual (Sample-wise)
plt.figure(figsize=(10, 5))
plt.plot(y_test, 'r--', label='Original Data')
plt.plot(y_pred_test, 'b-', label='QANFIS Output')
plt.xlabel('Number of Test Samples')
plt.ylabel('Flood Potential')
plt.title('QANFIS Output vs Actual (Sample-wise)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/qanfis_output_vs_actual_4qubit_data4.png", dpi=300)
plt.show()
