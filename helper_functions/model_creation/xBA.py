import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class xBAModel(nn.Module):
    def __init__(self, input_dim: int):
        super(xBAModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_xba_model(df: pd.DataFrame, features: list, epochs: int = 50, batch_size: int = 256):
    """
    Trains a PyTorch xBA neural network on a DataFrame pre-filtered for balls in play.
    
    Parameters:
        df (pd.DataFrame): Batted balls DataFrame containing feature columns and 'is_hit'.
        features (list): List of feature column names (e.g., ['launch_speed', 'launch_angle', ...]).
        epochs (int): Number of training epochs (default: 50).
        batch_size (int): Mini-batch size for training (default: 256).
        
    Returns:
        tuple: (model, scaler)
    """
    # 1. Drop rows with missing values in target or features
    clean_data = df.dropna(subset=features + ['is_hit']).copy()
    X = clean_data[features].values
    y = clean_data['is_hit'].values

    # 2. Train/Test Split & Feature Scaling
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Set compute device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Model & Optimization Setup
    model = xBAModel(input_dim=len(features)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. DataLoader Setup
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 6. Training Loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    print(f"Model training complete over {epochs} epochs on {device.type.upper()}.")
    return model, scaler

def predict_xba(model, scaler, df: pd.DataFrame, features: list, output_col: str = 'xBA') -> pd.DataFrame:
    """
    Predicts expected batting average (xBA) for a given DataFrame using a trained 
    PyTorch model and scaler. Handles dropping rows with missing feature values.

    Parameters:
        model (nn.Module): Trained PyTorch xBA model.
        scaler (StandardScaler): Fitted StandardScaler object.
        df (pd.DataFrame): DataFrame containing the required feature columns.
        features (list): List of feature column names used during training.
        output_col (str): Column name to store predictions (default: 'xBA').

    Returns:
        pd.DataFrame: DataFrame containing only clean rows with the added prediction column.
    """
    # 1. Clean missing values in the required features
    clean_df = df.dropna(subset=features).copy()
    
    # 2. Extract and scale feature values
    X_raw = clean_df[features].values
    X_scaled = scaler.transform(X_raw)
    
    # 3. Convert to PyTorch Tensor and detect compute device
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # 4. Perform inference
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()
        
    clean_df[output_col] = predictions
    return clean_df