#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import numpy as np
import pandas as pd
import time
from SlidingWindow import SlidingWindow
from LSTMAE import LSTMAE


def evaluate_buffer(model, buffer, device='cpu'):
    model.eval()
    
    error = buffer[:, 0]
    action = buffer[:, 1]
    error_rate = buffer[:, 2]
    
    buffer_seq = np.column_stack([error, action, error_rate])
    buffer_seq = buffer_seq.astype(np.float32)
    
    with torch.no_grad():
        buffer_tensor = torch.FloatTensor(buffer_seq).unsqueeze(0).to(device)
        reconstruction = model(buffer_tensor)
        mse = ((reconstruction - buffer_tensor) ** 2).mean()
        return mse.item()


def get_stats(training_file='dataset/Trainingset'):
    training_df = pd.read_csv(training_file, header=None)
    training_df[2] = training_df[2].astype(int)
    training_df = training_df.drop(training_df.columns[0], axis=1)
    training_set = training_df.to_numpy()
    
    window = SlidingWindow(window_size=60, step_size=1, buffer_size=20)
    
    rmse_values = []
    mae_values = []
    volatility_values = []
    temp_errors = []
    
    for data in training_set:
        window_data = window.add(data)
        
        if window_data is not None:
            current_win, rmse, mae, vol = window.current_window()
            rmse_values.append(rmse)
            mae_values.append(mae)
            volatility_values.append(vol)
            window_errors = window_data[:, 0]
            temp_errors.append(np.mean(window_errors))
    
    stats = {
        'rmse': {'99per': np.percentile(rmse_values, 99), '1per': np.percentile(rmse_values, 1)},
        'mae': {'99per': np.percentile(mae_values, 99), '1per': np.percentile(mae_values, 1)},
        'volatility': {'99per': np.percentile(volatility_values, 99)},
        'temp_error': {'99per': np.percentile(temp_errors, 99), '1per': np.percentile(temp_errors, 1)}
    }
    
    return stats


def compute_costs(model, dataset, stats=None, compare_exhaustive=False, device='cpu'):
    if stats is None:
        stats = get_stats()
    
    results = {}
    
    # Remove anomaly labels if present
    if dataset.shape[1] > 3:
        dataset = dataset[:, :-1]
    
    # --- SELECTIVE EVALUATION --- #
    window = SlidingWindow(window_size=40, step_size=1, buffer_size=30)
    
    selective_start = time.perf_counter()
    windows_evaluated = 0
    lstm_inference_time = 0
    
    for i, data in enumerate(dataset):
        window_data = window.add(data)
        
        if window_data is not None:
            current_window, rmse, mae, vol = window.current_window()
            
            # Check thresholds
            if not window.triggered:
                triggered = False
                
                if (data[0] > stats['temp_error']['99per'] or data[0] < stats['temp_error']['1per'] or
                    rmse > stats['rmse']['99per'] or rmse < stats['rmse']['1per'] or
                    mae > stats['mae']['99per'] or mae < stats['mae']['1per'] or
                    vol > stats['volatility']['99per']):
                    
                    triggered = True
                    window.start_buffer()
        
        # Evaluate buffer when ready
        if window.triggered and window.time_after_trigger >= (window.buffer_size - 1):
            buffer = window.get_buffer()
            
            if buffer is not None:
                lstm_start = time.perf_counter()
                evaluate_buffer(model, buffer, device)
                lstm_end = time.perf_counter()
                
                lstm_inference_time += (lstm_end - lstm_start)
                windows_evaluated += 1
    
    selective_total = time.perf_counter() - selective_start
    
    results['selective'] = {
        'windows_evaluated': windows_evaluated,
        'total_time': selective_total,
        'lstm_time': lstm_inference_time,
        'overhead_time': selective_total - lstm_inference_time,
        'data_examined_percentage': (windows_evaluated * 60 / len(dataset)) * 100 if windows_evaluated > 0 else 0
    }
    
    # --- EXHAUSTIVE EVALUATION (runs eval on all timesteps) --- #
    if compare_exhaustive:
        buffer_size = 60
        total_windows = len(dataset) - buffer_size + 1
        
        exhaustive_start = time.perf_counter()
        
        for i in range(total_windows):
            buffer = dataset[i:i + buffer_size, :3]
            
            with torch.no_grad():
                buffer_tensor = torch.FloatTensor(buffer).unsqueeze(0).to(device)
                reconstruction = model(buffer_tensor)
                mse = ((reconstruction - buffer_tensor) ** 2).mean()
        
        exhaustive_total = time.perf_counter() - exhaustive_start
        
        results['exhaustive'] = {
            'windows_evaluated': total_windows,
            'total_time': exhaustive_total
        }
        
        # Calculate savings
        results['savings'] = {'window_reduction': 1 - (results['selective']['windows_evaluated'] / total_windows)}
    
    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timeseries = ['abrupt_1', 'gradual_1', 'sensor_1', 'mixed_1']
    
    # Load model
    model = LSTMAE(input_size=3, hidden_size=64, seq_len=60, dropout=0.3)
    trained_params = torch.load('lstmae.pth', map_location=device)
    model.load_state_dict(trained_params['model_state_dict'])
    model = model.to(device)
    
    for ts in timeseries: 
        # Load test dataset
        df = pd.read_csv('dataset/'+ts)
        dataset = df.to_numpy()[:, 1:]  # Remove index column

        # Compare_exhaustive = True to compare selective and exhaustive, else False if not needed
        costs = compute_costs(model, dataset, compare_exhaustive=True, device=device)

        print("\n---", ts," Results ---")
        print(f"Selective: {costs['selective']['windows_evaluated']} windows")
        print(f"Total processing time: {costs['selective']['total_time']:.2f}s")
        print(f"LSTM inference: {costs['selective']['lstm_time']:.4f}s")
        print(f"Data examined: {costs['selective']['data_examined_percentage']:.1f}%")

        if 'exhaustive' in costs:
            print(f"\nExhaustive: {costs['exhaustive']['windows_evaluated']} windows")
            print(f"Total processing time: {costs['exhaustive']['total_time']:.2f}s")
            print(f"Window reduction: {costs['savings']['window_reduction']:.1%}")


# In[ ]:




