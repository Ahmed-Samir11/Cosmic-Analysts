import os
import torch
from obspy import read
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from pickle import dump
from tqdm import tqdm
# Function to extract features from a given trace
def extract_features(trace):
    data = trace.data
    
    # Statistical features
    mean_val = np.mean(data)
    std_val = np.std(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    
    # Frequency domain features
    fft_vals = fft(data)
    fft_magnitude = np.abs(fft_vals)
    fft_mean = np.mean(fft_magnitude)
    fft_std = np.std(fft_magnitude)
    
    # Signal energy
    energy = np.sum(data ** 2)
    
    features = {
        'mean': mean_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurt,
        'fft_mean': fft_mean,
        'fft_std': fft_std,
        'energy': energy
    }
    
    return features
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def save_pkl(obj, save_path):
    """Save a Pyleecan object in a pkl file using cloudpickle

    Parameters
    ----------
    obj: Pyleecan object
        object to save
    save_path: str
        file path
    """

    with open(save_path, "wb") as save_file:
        dump(obj, save_file)
def train_autoencoder(model, train_loader, val_loader, num_epochs, num_eval_epoch, patience, 
                      criterion=None, optimizer=None, scheduler=None, save_dir="", gpu_number=0):
    mkdir(save_dir)
    
    if criterion is None:
        criterion = nn.MSELoss()
    
    device = torch.device(f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loss = []
    val_loss = []
    val_reconstruction_errors = []
    val_anomalies_counts = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for _, (inputs, _)  in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss_epoch = running_loss / len(train_loader)
        train_loss.append(train_loss_epoch)
        
        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % num_eval_epoch == 0:
            result = evaluate_autoencoder(model, val_loader, criterion, device)
            val_loss.append(result["val_loss"])
            val_reconstruction_errors.append(result["val_reconstruction_error"])
            val_anomalies_counts.append(result["val_anomalies_count"])
            
            if result["val_loss"] < best_val_loss:
                best_val_loss = result["val_loss"]
                
                # Script and save the model for C++ inference
                scripted_model = torch.jit.script(model)
                scripted_model.save(os.path.join(save_dir, 'best_val_ckpt.pt'))
                
                torch.save({'model_ckpt': model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_val_loss": best_val_loss,
                            }, os.path.join(save_dir, 'best_val_ckpt.pth'))
                print(f"Best model saved at epoch {epoch + 1}, val loss: {best_val_loss}")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} consecutive evaluations.")
            
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    stats = {
        'train_loss': train_loss, 
        'val_loss': val_loss, 
        'val_reconstruction_errors': val_reconstruction_errors,
        'val_anomalies_counts': val_anomalies_counts
    }
    save_pkl(stats, os.path.join(save_dir, 'stats.pkl'))

    return stats

def evaluate_autoencoder(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    reconstruction_errors = []
    
    with torch.no_grad():
        for _, (inputs, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()
            reconstruction_error = torch.mean((inputs - outputs) ** 2, dim=1).cpu().numpy()
            reconstruction_errors.extend(reconstruction_error)
    
    val_loss /= len(dataloader)
    threshold = np.percentile(reconstruction_errors, 95)
    anomalies_count = np.sum(np.array(reconstruction_errors) > threshold)
    
    return {
        'val_loss': val_loss, 
        'val_reconstruction_error': reconstruction_errors, 
        'val_anomalies_count': anomalies_count
    }