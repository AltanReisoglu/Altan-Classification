import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
# Veri seti yükleme ve bölme
bc = datasets.load_iris()
x, y = bc.data, bc.target
x_train, x_valid, y_train, y_valid = train_test_split(
    torch.tensor(x, dtype=torch.float32),  # NumPy -> PyTorch Tensor
    torch.tensor(y, dtype=torch.long),     # NumPy -> PyTorch Tensor
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(x_train)
valid_scaled = scaler.transform(x_valid)
x_features,x_numerics=x_train.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_scaled = torch.tensor(train_scaled, dtype=torch.float32).to(device)
valid_scaled = torch.tensor(valid_scaled, dtype=torch.float32).to(device)
y_train = y_train.to(device)

class AltanClassifier(nn.Module):
    def __init__(self,x_numerics,hidden_Size):
        super(AltanClassifier,self).__init__()
        self.linear1 = nn.Linear(x_numerics, hidden_Size)
        self.linear2 = nn.Linear(hidden_Size, 128)

        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 3)  # Çıkış katmanı
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.loss=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.parameters(), lr=0.0001)
        self.batch_norm1 = nn.BatchNorm1d(hidden_Size)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
    def forward(self,x):
        out = self.linear1(x)
        out=self.batch_norm1(out)
        out = torch.relu(out)  # Aktivasyon fonksiyonu
        out = self.linear2(out)
        out=self.batch_norm2(out)
        out = torch.relu(out)  # Aktivasyon fonksiyonu
        out = self.linear3(out)
        out=self.batch_norm3(out)
        out = torch.relu(out)  # Aktivasyon fonksiyonu
        out = self.linear4(out)  # Sonuçlar (ör. sınıflar)
        return out
    def fit(self,x_train,y_true,epoch):
        for _ in range(epoch):
            y_pred=self.forward(x_train)
            loss=self.loss(y_pred,y_true)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if _ % 2 == 0:
                [w, b] = self.linear2.parameters()
                print(f"Epoch {_ + 1}: Weight = {w[0][0].item()}, Loss = {loss.item()}")
    def predict(self,x):
        model.eval()
        with torch.no_grad():
            y_predict=self.forward(x)
            return y_predict

model=AltanClassifier(x_numerics,256)
model = model.to(device)
model.fit(train_scaled,y_train,60)
pred_val=torch.argmax(model.predict(valid_scaled),dim=1).cpu().numpy()
print(accuracy_score(np.array(y_valid),pred_val))

