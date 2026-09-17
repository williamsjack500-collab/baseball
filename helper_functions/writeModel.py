import os
import torch
import joblib

def write_pytorch_model(model, scaler, subfolder: str, description: str = ""):
    """
    Saves a trained PyTorch xBA model, its fitted StandardScaler, and an optional 
    description text file into a specified subfolder under 'models/'.
    """
    # Define save path
    save_dir = os.path.join('models', subfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save PyTorch Model State Dict
    model_path = os.path.join(save_dir, f'xBA_pytorch_model_{subfolder}.pth')
    torch.save(model.state_dict(), model_path)
    
    # 2. Save Fitted StandardScaler
    scaler_path = os.path.join(save_dir, 'xBA_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # 3. Write Description / Metadata Text File
    if description:
        desc_path = os.path.join(save_dir, f'{subfolder}_description.txt')
        with open(desc_path, 'w') as f:
            f.write(description)
            
    print(f"Successfully saved model, scaler, and description to '{save_dir}/'")