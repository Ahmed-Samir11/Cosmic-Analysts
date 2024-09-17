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
def extract_features(trace, file_name):
    start_time = trace.stats.starttime
    end_time = trace.stats.endtime
    network =  trace.stats.network
    station = trace.stats.station
    channel = trace.stats.channel
    location =  trace.stats.location
    sampling_rate = trace.stats.sampling_rate
    samples_num = trace.stats.npts
    features = {
        'network' :  network,
        'station' : station,
        'channel' : channel,
        'location' :  location,
        'locID.CHA' : f'{location}.{channel}',
        'start_time': start_time,
        'end_time': end_time,
        'sampling_rate' : sampling_rate, 
        'num_of_samples' : samples_num,
        'file_name': file_name
        #'data': trace.data
}
  
    return features
def feature_engineering(trace, file_name):
    data = trace.data
    channel = trace.stats.channel
    location =  trace.stats.location
    data = data.astype(np.float64)
    # Statistical features
    mean_val = np.mean(data)
    std_val = np.std(data)
    skewness = skew(data, nan_policy='raise')
    kurt = kurtosis(data, nan_policy='raise')
    
    # Frequency domain features
    fft_vals = fft(data)
    fft_magnitude = np.abs(fft_vals)
    fft_mean = np.mean(fft_magnitude)
    fft_std = np.std(fft_magnitude)
    
    # Signal energy
    energy = np.sum(data ** 2)
    start_time = trace.stats.starttime
    end_time = trace.stats.endtime
    station = trace.stats.station
    sampling_rate = trace.stats.sampling_rate
    samples_num = trace.stats.npts
    features = {
        'file_name': file_name,
        'locID.CHA' : f'{location}.{channel}',
        'station' : station,
        'channel' : channel,
        'location' :  location,
        'start_time': start_time,
        'end_time': end_time,
        'sampling_rate' : sampling_rate, 
        'num_of_samples' : samples_num,
        'mean': mean_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurt,
        'fft_mean': fft_mean,
        'fft_std': fft_std,
        'energy': energy,
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
    threshold = np.percentile(reconstruction_errors, 99)
    anomalies_count = np.sum(np.array(reconstruction_errors) > threshold)
    
    return {
        'val_loss': val_loss, 
        'val_reconstruction_error': reconstruction_errors, 
        'val_anomalies_count': anomalies_count
    }
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
    threshold = np.percentile(reconstruction_errors, 99)
    anomalies_count = np.sum(np.array(reconstruction_errors) > threshold)
    
    return {
        'val_loss': val_loss, 
        'val_reconstruction_error': reconstruction_errors, 
        'val_anomalies_count': anomalies_count
    }
def train_no_val(model, train_loader, num_epochs, patience, 
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
    anomalies_counts = []
    best_train_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        reconstruction_errors = []
        
        for _, (inputs, _)  in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate reconstruction error
            reconstruction_error = torch.mean((inputs - outputs) ** 2, dim=1).detach().cpu().numpy()
            reconstruction_errors.extend(reconstruction_error)
        
        train_loss_epoch = running_loss / len(train_loader)
        train_loss.append(train_loss_epoch)
        
        # Calculate anomaly threshold and count anomalies
        threshold = np.percentile(reconstruction_errors, 99)  # 99th percentile
        anomalies_count = np.sum(np.array(reconstruction_errors) > threshold)
        anomalies_counts.append(anomalies_count)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_epoch}, Anomalies: {anomalies_count}")
        
        if scheduler is not None:
            scheduler.step()
        
        # Check if the current training loss is the best
        if train_loss_epoch < best_train_loss:
            best_train_loss = train_loss_epoch
            torch.save({'model_ckpt': model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_train_loss": best_train_loss,
                        }, os.path.join(save_dir, 'best_train_ckpt.pth'))
            print(f"Best model saved at epoch {epoch + 1}, train loss: {best_train_loss}")
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1
            print(f"No improvement in training loss for {patience_counter} consecutive epochs.")
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    stats = {
        'train_loss': train_loss,
        'anomalies_counts': anomalies_counts
    }
    save_pkl(stats, os.path.join(save_dir, 'stats.pkl'))

    return stats
