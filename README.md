# Digital Twin Fidelity Evaluation Framework
A runtime anomaly detection framework for evaluating digital twin fidelity using sliding windows and LSTM Autoencoder.
### Overview:
This framework distinguishes between expected inaccuracies in low-fidelity digital twins and genuine anomalous behaviour through a two-stage detection pipeline of statistical pre-filtering with sliding windows and deep learning validation with LSTM-AE.

### Includes:
+ IncSim.py - Incubator simulator with configurable anomaly scenarios 
+ LSTMAE.py - LSTM Autoencoder architecture 
+ SlidingWindow.py - Sliding window algorithm with window based metric calculations
+ trainingscript.py - Model training script with chosen hyperparameters
+ Main.py - model evaluations and visualisation
+ datasets- Training and test datasets 
+ lstmae.pth - Trained model parameters
+ results - evaluated anomaly timeseries result plots and confusion matrices
