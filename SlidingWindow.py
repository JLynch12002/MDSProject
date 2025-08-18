#!/usr/bin/env python
# coding: utf-8

# In[10]:


from collections import deque
import numpy as np
import pandas as pd
import random
import math

class SlidingWindow:
    def __init__(self, window_size, step_size, buffer_size):
        self.window_size = window_size
        self.step_size = step_size
        self.window = deque(maxlen=window_size)
        self.step_count = 0
        
        # Buffer 
        self.buffer_size = buffer_size
        self.buffer = []
        self.triggered = False
        self.data_history = deque(maxlen=(buffer_size + 1))  
        self.time_after_trigger = 0
        
    def add(self, data_point):
        # Always maintain recent history for buffer when triggered
        self.data_history.append(data_point)
        
        # Adds new time step to the window and removes the oldest
        self.window.append(data_point)
        self.step_count += 1
        
        
        # If collecting, add current point to buffer
        if self.triggered:
            self.add_to_buffer(data_point)
        
        if len(self.window) == self.window_size and self.step_count >= self.step_size:
            self.step_count = 0
            return np.array(self.window)
    
        
    def start_buffer(self):
        # Starts collecting buffer
        if not self.triggered:
            self.triggered = True
            self.time_after_trigger = 0
            # creates buffer with previous t data points (excluding current)
            self.buffer = list(self.data_history)[:-1]
                
    def add_to_buffer(self, data_point):
        # Adds current data point to buffer
        self.buffer.append(data_point)
        self.time_after_trigger += 1
        
        # Stop if collected enough points after trigger
        if self.time_after_trigger >= self.buffer_size:
            self.triggered = False
            return self.get_buffer()
    
    def get_buffer(self):
        # returns the buffer as a single array
        if self.buffer:
            return np.array(self.buffer)
 
    def current_window(self):
        # The current window, calculates window metrics
        if len(self.window) == self.window_size:
            rmse = self.RMSE(self.window)  
            mae = self.MAE(self.window)
            vol = self.volatility(self.window)
            return np.array(self.window), rmse, mae, vol
            
    def reset(self):
        # Clears the window and buffer
        self.window.clear()
        self.step_count = 0
        self.buffer = []
        self.triggered = False
        self.data_history.clear()
        self.points_after_trigger = 0
    
    def RMSE(self, window):
        # Calculates the RMSE of each window 
        error = [] 
        for i in window:
            error.append(i[0])
        pred_error = np.array(error)
        return np.sqrt(np.mean((pred_error)**2))
        
    def MAE(self, window):
        # Calculates the MAE of each window 
        error = [] 
        for i in window:
            error.append(i[0])
        pred_error = np.array(error)
        return np.mean(np.abs(pred_error))
        
    def volatility(self, window):
        # Standard deviation of 5 most recent absolute errors 
        errors = [abs(i[0]) for i in window[-5:]]
        return np.std(errors)
    

