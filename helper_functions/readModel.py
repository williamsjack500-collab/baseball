import os
import torch
import joblib

def read_pytorch_model(subfolder: str):
    """
    Loads and returns the PyTorch model state_dict and fitted StandardScaler 
    from a specified subfolder under 'models/'.
    
    Parameters:
        subfolder (str): The version or directory name inside 'models/' (e.g., 'v1').
        
    Returns:
        tuple: (state_dict, scaler)
    """
    model_dir = os.path.join('models', subfolder)
    
    model_path = os.path.join(model_dir, f'xBA_pytorch_model_{subfolder}.pth')
    scaler_path = os.path.join(model_dir, 'xBA_scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at '{scaler_path}'")
        
    # Load PyTorch state dict and scaler
    state_dict = torch.load(model_path, weights_only=True)
    scaler = joblib.load(scaler_path)
    
    print(f"Successfully loaded model weights and scaler from '{model_dir}/'")
    return state_dict, scaler