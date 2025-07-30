# --- QANFIS dengan PauliFeatureMap  ---

import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit

# 1. Reproducibility Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Dataset
df = pd.read_csv("dummy_flood_data.csv")
X = df[['rainfall', 'water_level']].values
y = df['flood_potential'].values.reshape(-1, 1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y).flatten()

X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=SEED)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# 3. Trainable Gaussian MF
class TrainableGaussianMF(nn.Module):
    def __init__(self, num_mf):
        super().__init__()
        self.mu = nn.Parameter(torch.linspace(0, 1, num_mf))
        self.sigma = nn.Parameter(torch.ones(num_mf) * 0.2)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

# 4. Quantum Circuit using PauliFeatureMap
num_mf = 3
num_rules = num_mf ** 2

feature_map = PauliFeatureMap(feature_dimension=num_rules, reps=1, paulis=['X', 'Y', 'Z'])
ansatz = RealAmplitudes(num_qubits=num_rules, reps=2, entanglement='linear')

qc = QuantumCircuit(num_rules)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

qnn = SamplerQNN(circuit=qc,
                 input_params=feature_map.parameters,
                 weight_params=ansatz.parameters,
                 sampler=Sampler())

torch_qnn = TorchConnector(qnn).to(device)

# 5. QANFIS Model
class QANFISNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_mf = num_mf
        self.num_rules = num_rules
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
        bit_weights = torch.linspace(0, 1, outputs.shape[1], device=outputs.device)
        expectation = torch.sum(outputs * bit_weights, dim=1)
        return expectation

# 6. Training Setup
model = QANFISNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

best_val_loss = float('inf')
patience, patience_counter = 10, 0
train_losses, val_losses = [], []

# 7. Training Loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = criterion(y_val_pred, y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# 8. Evaluation
model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).cpu().numpy()
    y_true_test = y_test

rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
r2 = r2_score(y_true_test, y_pred_test)

# 9. Plot Results
plt.figure(figsize=(8,6))
plt.scatter(y_true_test, y_pred_test, alpha=0.6, color="green")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Flood Potential")
plt.ylabel("Predicted (QANFIS)")
plt.title(f"QANFIS Regression Result\nRMSE: {rmse:.4f}, R²: {r2:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/qanfis_pauli_scatter.png", dpi=300)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QANFIS Training & Validation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output/qanfis_pauli_loss.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_true_test, 'r--', label='Original Data')
plt.plot(y_pred_test, 'b-', label='QANFIS Output')
plt.xlabel('Number of Test Samples')
plt.ylabel('Flood Potential')
plt.title('QANFIS Output vs Actual (Pauli Feature Map)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/qanfis_pauli_vs_actual.png", dpi=300)
plt.show()
