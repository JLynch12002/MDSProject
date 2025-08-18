#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np
import pandas as pd
from LSTMAE import LSTMAE
from SlidingWindow import SlidingWindow
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sn
import scipy.stats as scpy
import statsmodels.api as sm



# Passes buffer window through LSTM-AE, returns reconstruction error 
def evaluate_buffer(model, buffer, device='cpu'):

    model.eval()
    
    error = buffer[:, 0]
    action = buffer[:, 1]
    error_rate = buffer[:, 2]
    
    # Stack all features together
    buffer_seq = np.column_stack([error, action, error_rate])
    buffer_seq = buffer_seq.astype(np.float32)
    
    with torch.no_grad():
        # Convert to tensor and add batch dimension ([seq_len, 3] to [1, seq_len, 3])
        buffer_tensor = torch.FloatTensor(buffer_seq).unsqueeze(0).to(device)
        
        # reconstruction error 
        reconstruction = model(buffer_tensor)
        
        # Calculate MSE of the reconstructed window 
        mse = ((reconstruction - buffer_tensor) ** 2).mean()
        
        return mse.item()
    
    
    
    
# Calculates the mean and SD of training dataset window's rmse, mae, volatility and pred temp error 
def metric_stats(training_data, window_size, step_size):
    
    window = SlidingWindow(window_size=window_size, step_size=step_size, buffer_size=20) # ignore buffer size, not used here
    
    # stat values
    rmse_values = []
    mae_values = []
    volatility_values = []
    temp_errors = []
    
    # Processes training set through sliding windows 
    for data in training_data:
        window_data = window.add(data)
        
        if window_data is not None:
            current_win, rmse, mae, vol = window.current_window()
            rmse_values.append(rmse)
            mae_values.append(mae)
            volatility_values.append(vol)
            
            # Calculate mean temperature error for this window
            window_errors = window_data[:, 0]
            temp_errors.append(np.mean(window_errors))
    
    # Calculate window percentile statistics for each metric
    stats = {
        'rmse': {
            'mean': np.mean(rmse_values),
            'std': np.std(rmse_values),
            '99per': np.percentile(rmse_values, 99),
            '1per': np.percentile(rmse_values, 1)},
        'mae': {
            'mean': np.mean(mae_values),
            'std': np.std(mae_values),
            '99per': np.percentile(mae_values, 99),
            '1per': np.percentile(mae_values, 1)},
        'volatility': {
            'mean': np.mean(volatility_values),
            'std': np.std(volatility_values),
            '99per': np.percentile(volatility_values, 99)},
        'temp_error': {
            'mean': np.mean(temp_errors),
            'std': np.std(temp_errors),
            '99per': np.percentile(temp_errors, 99),
            '1per': np.percentile(temp_errors, 1)}}
    

    return stats, rmse_values, mae_values, volatility_values, temp_errors


# In[3]:


# Loading training datset for normal behaviour metric stats calculation
training_df = pd.read_csv('dataset/Trainingset', header=None)
training_df[2] = training_df[2].astype(int)
training_df = training_df.drop(training_df.columns[0], axis=1) # removes index
training_set = training_df.to_numpy()


stats, rmse_values, mae_values, volatility_values, temp_errors = metric_stats(training_set, 30, 1)


# In[4]:


# --- plots the metrics --- #
# while the distributions of the metrics aren't necessary for the model, it helped to decide the type of thresholds employed
metrics = {"rmse": rmse_values,
            "mae": mae_values,
            "volatility": volatility_values,
            "temp_error": temp_errors }

num_metrics = len(metrics)
fig, axs = plt.subplots(num_metrics, 3, figsize=(12, 4 * num_metrics))

for i, (name, values) in enumerate(metrics.items()):
    
    axs[i, 0].hist(values, bins=50, density=True)
    axs[i, 0].axvline(stats[name]['99per'], color='r')
    if '1per' in stats[name]:
        axs[i, 0].axvline(stats[name]['1per'], color='r')
    axs[i, 0].set_title(f"{name} Histogram")
    
    skew = scpy.skew(values)
    kurt = scpy.kurtosis(values)
    axs[i, 0].text(0.05, 0.9, f"Skewness: {skew:.4f}", transform=axs[i, 0].transAxes, fontsize=12)
    axs[i, 0].text(0.05, 0.8, f"Kurtosis: {kurt:.4f}", transform=axs[i, 0].transAxes, fontsize=12)
    
    sm.qqplot(np.array(values), line='45', ax=axs[i,1])
    axs[i, 1].set_title(f"{name} Q-Q Plot")

    axs[i, 2].violinplot(values, showextrema=True, showmedians=True)
    axs[i, 2].set_title(f"{name} Violin plots")

plt.tight_layout()
plt.show()


# In[12]:


# ---- Evaluates and plots the timeseries ---- # 

if __name__ == "__main__":
    display_stats = False
    testsets = ['mixed_1', 'mixed_2', 'mixed_3', 'mixed_4', 'mixed_5'] # change the timeseries names here
    w_acc_scores = []
    w_recall_scores = []
    w_precision_scores = []
    w_f1_scores = []
    p_acc_scores = []
    p_recall_scores = []
    p_precision_scores = []
    p_f1_scores = []
    eva_timesteps = []
    
    
    
    
    for timeseries in testsets:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialise model 
        model = LSTMAE(input_size=3, hidden_size=64, seq_len=60, dropout=0.3)

        # Load trained parameters
        trained_params = torch.load('lstmae.pth', map_location=device)
        model.load_state_dict(trained_params['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Initalise sliding window
        window = SlidingWindow(window_size=40, step_size=1, buffer_size=30)

        # Load anomaly dataset
        file = "dataset/"+timeseries
        df = pd.read_csv(file)
        df = df.drop(df.columns[0], axis=1)
        dataset = df.to_numpy()

        # Seperates the anomaly labels from dataset
        anomaly_labels = dataset[:,-1]
        dataset = dataset[:, :-1]

        # Logs window and model activity 
        trigger_times = [] # stores times when triggered
        prefilter_triggers = [] # has the percentile threshold be breached 
        confirmed_anomalies = [] # stores times evaluated as anomalous 
        recon_scores = [] 
        evaluated_windows = []  # Tracks all evaluated windows 


        ######### run-time simulation (data points added to window) ##################
        for time, data in enumerate(dataset):
            window_data = window.add(data)

            if window_data is not None:
                current_window, rmse, mae, vol = window.current_window()

                # Collects whether sliding window thresholds been triggered (for pre-filter evalaution)(incredibly ugly but most simple way)
                if data[0] > stats['temp_error']['99per'] or data[0] < stats['temp_error']['1per'] or rmse > stats['rmse']['99per'] or rmse < stats['rmse']['1per'] or mae > stats['mae']['99per'] or mae < stats['mae']['1per'] or  vol > stats['volatility']['99per']:
                    prefilter_triggers.append(1)
                else:
                    prefilter_triggers.append(0)

            if not window.triggered and window_data is not None:
                triggered = False
                broken_thres = []

                # Thresholds
                # (temp error > 0 as 0 error ideal)
                # (Vol cannot go below 0) 
                if data[0] > stats['temp_error']['99per'] or data[0] < stats['temp_error']['1per']:
                    triggered = True
                    broken_thres.append('temp error')
                if rmse > stats['rmse']['99per'] or rmse < stats['rmse']['1per']:
                    triggered = True
                    broken_thres.append('rmse')
                if mae > stats['mae']['99per'] or mae < stats['mae']['1per']:
                    triggered = True
                    broken_thres.append('mae')
                if vol > stats['volatility']['99per']:
                    triggered = True
                    broken_thres.append('vol')

                if triggered:
                    window.start_buffer()
                    trigger_times.append(time)

            # Checks the buffer is full before passing it to LSTM-AE    
            if window.triggered and window.time_after_trigger >= (window.buffer_size-1):
                buffer = window.get_buffer()

                if buffer is not None:
                    reconstruction_error = evaluate_buffer(model, buffer, device)
                    recon_scores.append(reconstruction_error)

                    # Calculate window boundaries
                    window_start = trigger_times[-1] - window.buffer_size
                    window_end = trigger_times[-1] + (window.buffer_size - 1)

                    evaluated_windows.append((window_start, window_end, reconstruction_error))

                    # mean reconstruction error on normal val results mean = 0.013, max approx 0.0607
                    if reconstruction_error >= 0.15:  
                        # Store the confirmed anomaly window times
                        confirmed_anomalies.append((window_start, window_end, trigger_times[-1]))


        ############# Model evaluation ##################### 

        ### Window-level evaluation
        window_truth = []
        window_preds = []

        for window_start, window_end, recon_error in evaluated_windows:
            start_ind = max(0, window_start)
            end_ind = min(len(dataset), window_end + 1)

            # Checks if evaluated window contains anomalies 
            contains_anomaly = np.any(anomaly_labels[start_ind:end_ind] == 1)
            window_truth.append(int(contains_anomaly))

            # Checks if any evaluated windows break recon threshold
            predicted_anomaly = int(recon_error >= 0.15)
            window_preds.append(predicted_anomaly)

        # Calculate window level metrics
        if len(confirmed_anomalies) > 0:
            window_accuracy = accuracy_score(window_truth, window_preds)
            window_precision = precision_score(window_truth, window_preds, zero_division=0)
            window_recall = recall_score(window_truth, window_preds, zero_division=0)
            window_f1 = f1_score(window_truth, window_preds, zero_division=0)
            window_cm = confusion_matrix(window_truth, window_preds)
            
            w_acc_scores.append(window_accuracy)
            w_recall_scores.append(window_recall)
            w_precision_scores.append(window_precision)
            w_f1_scores.append(window_f1)



        ### Point-level evaluation

        evaluated_mask = np.zeros(len(dataset), dtype=bool)
        pred_labels = np.zeros(len(dataset))

        for window_start, window_end, recon_error in evaluated_windows:
            start_ind = max(0, window_start)
            end_ind = min(len(dataset), window_end + 1)
            evaluated_mask[start_ind:end_ind] = True

            if recon_error >= 0.15:
                pred_labels[start_ind:end_ind] = 1

        # Point-level metrics on evaluated regions only
        if np.sum(confirmed_anomalies) > 0:
            point_accuracy = accuracy_score(anomaly_labels[evaluated_mask], pred_labels[evaluated_mask])
            point_precision = precision_score(anomaly_labels[evaluated_mask], pred_labels[evaluated_mask], zero_division=0)
            point_recall = recall_score(anomaly_labels[evaluated_mask], pred_labels[evaluated_mask], zero_division=0)
            point_f1 = f1_score(anomaly_labels[evaluated_mask], pred_labels[evaluated_mask], zero_division=0)
            point_cm = confusion_matrix(anomaly_labels[evaluated_mask], pred_labels[evaluated_mask])
            
            p_acc_scores.append(point_accuracy)
            p_recall_scores.append(point_recall)
            p_precision_scores.append(point_precision)
            p_f1_scores.append(point_f1)


        ################### Stat trigger evaluation  #################

        trigger_predictions = np.zeros(len(dataset))
        for trigger_time in trigger_times:
            trigger_predictions[trigger_time] = 1

        prefilter_triggers = np.array(prefilter_triggers)
        # Done to align list lengths (Anomalies don't accur for intial 30sec duration so shouldn't effect results)
        aligned_anomaly_labels = np.array(anomaly_labels[-len(prefilter_triggers):])

        prefilter_precision = precision_score(aligned_anomaly_labels, prefilter_triggers, zero_division=0)
        prefilter_recall = recall_score(aligned_anomaly_labels, prefilter_triggers, zero_division=0)

        # Separate true/false positives for plotting
        true_pos_triggers = [t for t in trigger_times if anomaly_labels[t] == 1]
        false_pos_triggers = [t for t in trigger_times if anomaly_labels[t] == 0]

        ################# Plotting evaluations ###################

        if len(confirmed_anomalies) > 0:
            if display_stats == True:
                print(f"\n---- Window-Level Performance ----")
                print(f"Windows evaluated: {len(evaluated_windows)}")
                print(f"Windows predicted as anomalies: {len(confirmed_anomalies)}")
                print(f"Accuracy: {window_accuracy:.2%}")
                print(f"Precision: {window_precision:.2%}")
                print(f"Recall: {window_recall:.2%}")
                print(f"F1-Score: {window_f1:.2%}")

                print(f"\n---- Point-Level Performance ----")
                print(f"Total time steps: {len(dataset)}")
                print(f"Evaluated time steps: {np.sum(evaluated_mask)} ({np.sum(evaluated_mask)/len(dataset):.2%})")
                print(f"Accuracy: {point_accuracy:.2%}")
                print(f"Precision: {point_precision:.2%}")
                print(f"Recall: {point_recall:.2%}")
                print(f"F1-Score: {point_f1:.2%}")
                
                
            
            eva_timesteps.append(np.sum(evaluated_mask))
            
            # Confusion matrix plot using window metrics
            sn.heatmap(window_cm, annot=True, fmt=',d',cmap='Blues',xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
            plt.ylabel('Ground Truth', fontweight='bold')  # Bold label
            plt.xlabel('Model Prediction', fontweight='bold')  # Bold label
            #print(f"\nWindow Confusion Matrix:")
            #plt.savefig('results/'+timeseries+'_cm.png')
            plt.show()

            plt.figure(figsize = (14, 8))
            plt.plot(range(len(dataset)), dataset[:, 0], 'b-', label='Temp Error')

            # Shade regions where ground truth anomalies occur
            for i in range(len(anomaly_labels)):
                if anomaly_labels[i] == 1:
                    if i == 0 or anomaly_labels[i-1] == 0:
                        start = i
                    if i == len(anomaly_labels)-1 or anomaly_labels[i+1] == 0:
                        plt.axvspan(start, i, alpha=0.3, color='red', hatch ='///', label='Actual Anomaly' if 'Actual Anomaly' not in plt.gca().get_legend_handles_labels()[1] else '')

            # Shade regions where model predicts anomalies e
            for i in range(len(pred_labels)):
                if pred_labels[i] == 1:
                    if i == 0 or pred_labels[i-1] == 0:
                        start = i
                    if i == len(pred_labels)-1 or pred_labels[i+1] == 0:
                        plt.axvspan(start, i, alpha=0.3, color='green', label='Predicted Anomaly' if 'Predicted Anomaly' not in plt.gca().get_legend_handles_labels()[1] else '')

            plt.xlabel('Time Step', fontweight='bold') 
            plt.ylabel('Temperature Error', fontweight='bold')  
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            #plt.savefig('results/'+timeseries+'_plot.png')
            plt.show()

        else:
            print(f"--- No Anomalies Detected ---")


        print(f"\n---- Trigger Summary ----")
        print(f"Threshold breaches: {len(trigger_times)}")
        print(f"Total predicted anomalies: {len(confirmed_anomalies)}")
        print(f"\nPre-filter Precision: {prefilter_precision:.2%}")
        print(f"Pre-filter Recall: {prefilter_recall:.2%}\n")

        # Plot temp error with trigger times (quoted out to save processing time, takes long for longer timeseries)
        '''
        plt.figure(figsize = (14, 8))
        plt.plot(range(len(dataset)), dataset[:, 0], 'b-', label='Temp Error')

        for t in range(len(prefilter_triggers)):
            if prefilter_triggers[t] == 1:
                if anomaly_labels[t] == 1:
                    plt.axvline(x=t, color='green',  linestyle='--', linewidth=2, alpha=0.8, label='Prefilter TP' if 'Prefilter TP' not in plt.gca().get_legend_handles_labels()[1] else '')
                else:
                    plt.axvline(x=t, color='red', linestyle='--', linewidth=0.5, alpha=0.7, label='Prefilter FP' if 'Prefilter FP' not in plt.gca().get_legend_handles_labels()[1] else '')
        plt.xlabel('Time Step')
        plt.ylabel('Temp Error')
        plt.title('Triggers Outcomes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('normal_test_trigger.png')
        plt.show()
        
        '''

    
print(f" ---- Window Evaluation ----")
print(f'Evaluated timesteps: {sum(eva_timesteps)}')

print(f'Mean Window Accuracy: {np.mean(w_acc_scores):.3f}')
print(f'Accuracy Min: {np.min(w_acc_scores):.3f}')
print(f'Accuracy Max: {np.max(w_acc_scores):.3f}')

print(f'Mean Window Recall: {np.mean(w_recall_scores):.3f}')
print(f'Recall Min: {np.min(w_recall_scores):.3f}')
print(f'Recall Max: {np.max(w_recall_scores):.3f}')

print(f'Mean Window Precision: {np.mean(w_precision_scores):.3f}')
print(f'Precision Min: {np.min(w_precision_scores):.3f}')
print(f'Precision Max: {np.max(w_precision_scores):.3f}')

print(f'Mean Window F1: {np.mean(w_f1_scores):.3f}')
print(f'F1 Min: {np.min(w_f1_scores):.3f}')
print(f'F1 Max: {np.max(w_f1_scores):.3f}')

print(f" ---- Point Evaluation ----")

print(f'Mean Point Accuracy: {np.mean(p_acc_scores):.3f}')
print(f'Accuracy Min: {np.min(p_acc_scores):.3f}')
print(f'Accuracy Max: {np.max(p_acc_scores):.3f}')

print(f'Mean Point Recall: {np.mean(p_recall_scores):.3f}')
print(f'Recall Min: {np.min(p_recall_scores):.3f}')
print(f'Recall Max: {np.max(p_recall_scores):.3f}')

print(f'Mean Point Precision: {np.mean(p_precision_scores):.3f}')
print(f'Precision Min: {np.min(p_precision_scores):.3f}')
print(f'Precision Max: {np.max(p_precision_scores):.3f}')

print(f'Mean Point F1: {np.mean(p_f1_scores):.3f}')
print(f'F1 Min: {np.min(p_f1_scores):.3f}')
print(f'F1 Max: {np.max(p_f1_scores):.3f}')

