#!/usr/bin/env python
# coding: utf-8

# In[201]:


import simpy
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import random

random.seed(10)
np.random.seed(10)

class Incubator:
    def __init__(self, env, starting_temp, duration, config):
        self.env = env
        self.actual_temp = starting_temp
        self.configuration = config
        self.duration = duration
        self.starting_temp = starting_temp
        
        # Heater properties
        self.heater_on = False # heater starts turned off 
        self.heater_power = 0 # Heater starts at no power
        self.heater_max_rate = 0.5  # Max heat produced when heater full power 
        self.heater_ramp_time = 10  # Time to reach full power 
        self.heater_command_time = None  # Time when heater command was issued
        self.heater_target_state = False  # Target state after latency
        
        # Anomaly behaviour
        self.active_anomaly = False 
        self.anomaly_history = []
        
        # Heat transfer parameters
        self.heat_loss_coefficient = 0.02  # 0.02 representing acceptable insulation
        
        # Logs activity & temps
        self.temp_history = []
        self.sensor_history = []
        self.heater_history = []
        self.time_history = []
        self.heater_temp_history = []

        # Logs anomaly occurances 
        self.anomalies = []  
        self.config_anomalies()
        
        
    def config_anomalies(self):
        # Config 1: gradual environment drifts
        if self.configuration == 1:
            num_anomaly = 1
            placed_anomalies = 0 # Issue with while loop creating more anomalies than wanted. Hard coded max to break loop
            max_attempts = 10
            cooldown = 40  # minimum timesteps between next drift occuring 

            while placed_anomalies < num_anomaly:
                drift_attempts = 0
                while drift_attempts < max_attempts:
                    drift_attempts += 1
                    anomaly_start = random.randint(30, self.duration)
                    drift_duration = random.randint(400, 1000)

                    # Prevents overlapping drifts
                    if (anomaly_start + drift_duration) > self.duration:
                        drift_duration = self.duration - anomaly_start

                    drift_end = anomaly_start + drift_duration

                    clash = False
                    for anomaly in self.anomalies:
                        if anomaly['type'] == 'gradual':
                            anomaly_buffer_start = anomaly['start'] - cooldown
                            anomaly_buffer_end = (anomaly['start'] + anomaly['duration']) + cooldown
                            # Check overlap clash
                            if not (drift_end < anomaly_buffer_start or anomaly_start > anomaly_buffer_end):
                                clash = True
                                break

                    if not clash:
                        self.anomalies.append({'type': 'gradual',
                                                'start': anomaly_start,
                                                'duration': drift_duration,
                                                'magnitude': 0.15})
                        placed_anomalies += 1
                        break  
                    
                    
        # Config 2: sensor interference anom
        elif self.configuration == 2:
            num_anomaly = 5
            for i in range(num_anomaly):
                anomaly_time = random.randint(30, self.duration)
                self.anomalies.append({'type': 'abrupt',
                                        'start': anomaly_time,
                                        'duration': 1,  
                                        'magnitude': random.randint(400, 500)})
                
                

        # Config 3: sensor failure anom
        elif self.configuration == 3:
            num_anomaly = 2
            placed_anomalies = 0 
            max_attempts = 10
            cooldown = 100 
            
            while placed_anomalies < num_anomaly:
                drift_attempts = 0
                while drift_attempts < max_attempts:
                    drift_attempts += 10
                    anomaly_start = random.randint(30, self.duration)
                    drift_duration = random.randint(40, 80)

                    if (anomaly_start + drift_duration) > self.duration:
                        drift_duration = self.duration - anomaly_start

                    drift_end = anomaly_start + drift_duration

                    clash = False
                    for anomaly in self.anomalies:
                        if anomaly['type'] == 'sensor fail':
                            anomaly_buffer_start = anomaly['start'] - cooldown
                            anomaly_buffer_end = (anomaly['start'] + anomaly['duration']) + cooldownh
                            if not (drift_end < anomaly_buffer_start or anomaly_start > anomaly_buffer_end):
                                clash = True
                                break

                    if not clash:
                        self.anomalies.append({'type': 'sensor fail',
                                                'start': anomaly_start,
                                                'duration': drift_duration})
                        placed_anomalies += 1
                        break  
            
        
            
        # Config 4: Mixed gradual and abrupt drifts 
        elif self.configuration == 4:
            
            # Calculate graduals
            num_gradual = 1 
            for i in range(num_gradual):
                anomaly_start = random.randint(30, self.duration)
                drift_duration = random.randint(400, 1000)
                # Checks drift duration is not greater than sim duration
                if (anomaly_start + drift_duration) > self.duration:
                    drift_duration = self.duration - anomaly_start
                self.anomalies.append({'type': 'gradual',
                                       'start': anomaly_start,
                                       'end': anomaly_start + drift_duration,
                                       'duration': drift_duration,
                                       'magnitude': 0.15})
                
            # Calculate abrupts
            num_abrupt = 5
            abrupt_attempts = 0
            max_attempts = 100  
            
            # Checks to ensure abrupt drifts don't occur during gradual drifts
            while len([a for a in self.anomalies if a['type'] == 'abrupt']) < num_abrupt and abrupt_attempts < max_attempts:
                abrupt_attempts += 1
                anomaly_time = random.randint(30, self.duration)
                
                clash = False
                for anomaly in self.anomalies:
                    if anomaly['type'] == 'gradual':
                        # Can't occur 40 time steps before or after gradual (gives temp error time to recover to normal ops)
                        if anomaly_time >= (anomaly['start'] - 40) and anomaly_time <= (anomaly['end'] + 40):
                            clash = True
                            break               
                if not clash:
                        self.anomalies.append({'type': 'abrupt',
                                                'start': anomaly_time,
                                                'duration': 1,  
                                                'magnitude': random.randint(400, 500)})
  
                
    def check_anomaly(self, anomaly_type):
        # Checks if anomaly is active at current env time
        for anomaly in self.anomalies:
            if anomaly['type'] == anomaly_type:
                if self.env.now >= anomaly['start'] and self.env.now < (anomaly['start'] + anomaly['duration']):
                    return anomaly
                
                # For abrupt anomalies also considers a window around the spike as anomalous to avoid surround time steps captured in buffer to be classed false positive 
                # --- Not included in final testing ----
                elif anomaly['type'] == 'abrupt':
                    # Change values to marks x previous and x upcoming time steps as anom 
                    window_before = 0
                    window_after = 0
                    if self.env.now >= (anomaly['start'] - window_before) and self.env.now <= (anomaly['start'] + window_after):
                        # Return a modified anomaly dict to indicate this is within the anomaly window
                        # Actual spike mag not effected (magnitude will be 0 for non-spike timesteps)
                        return {'type': 'abrupt_window',
                                'start': anomaly['start'],
                                'duration': anomaly['duration'],
                                'magnitude': 0}
                    
        
    def set_heater(self, state, latency):
        # Set heater state with latency
        self.heater_command_time = self.env.now + latency
        self.heater_target_state = state
        
    def update_heater(self):
        # Checks if command has been issued and enough time has passed with added latency
        if self.heater_command_time is not None and self.env.now >= self.heater_command_time:
            self.heater_on = self.heater_target_state
            # Resets command time
            self.heater_command_time = None
            
        # Updates heater power 
        if self.heater_on is True:
            # 1 to ensure heat power never exceeds max heating rate
            self.heater_power = min((self.heater_power + (1.0 / self.heater_ramp_time)), 1)
        else:
            self.heater_power = 0
            
    def get_sensor_reading(self):
        # Adds random noise to actual temp to replicate sensors (sensor precision ± 3C) 
        noise = np.random.normal(0, 3)
        
        # Check for abrupt anomaly
        abrupt_anomaly = self.check_anomaly('abrupt')
        if abrupt_anomaly:
            noise += abrupt_anomaly['magnitude']

        # Config 3 'turns off' sensor (sensor temp = 0)
        failure_anomaly = self.check_anomaly('sensor fail')
        if failure_anomaly:
            time_elapsed = self.env.now - failure_anomaly['start']
            progress = time_elapsed / failure_anomaly ['duration']
            
            # sensor begins to resume after 0.5 of the anomaly activating (helps partly account for lasting effects) 
            if progress < 0.5:
                return 0 
            else:
                return self.actual_temp + noise
            
        return self.actual_temp + noise
    
    def update_temperature(self):
        # Updates actual temp based on model 
        self.update_heater()
        
        # Config 1 adds gradual drift (represent opening in incubator)
        gradual_anomaly = self.check_anomaly('gradual')
        if gradual_anomaly:
            # checks if duration is midway through before removing additional heat loss
            time_elapsed = self.env.now - gradual_anomaly['start']
            progress = time_elapsed / gradual_anomaly['duration']
        
        # Only apply heat loss during first 70% of anomaly duration
            if progress < 0.7:
                gradual_heat_loss = gradual_anomaly['magnitude']
            else:
                gradual_heat_loss = 0
        else:
            gradual_heat_loss = 0
            
            
        # Heat gain from heater with added random variation
        heat_gain = (self.heater_power * self.heater_max_rate)
        if self.heater_on:
            heat_gain_var = np.random.normal(0, 0.1) 
            # ensures heat gain doesn't exceed heater's max rate
            if (heat_gain_var + heat_gain) <= self.heater_max_rate:
                heat_gain += heat_gain_var
        heater_temp = heat_gain
            
        
        # Heat loss to environment with random variation and gradual heat loss if config 1
        heat_loss_variation = np.random.uniform(0, 0.05)  
        heat_loss = self.heat_loss_coefficient + heat_loss_variation + gradual_heat_loss
        
        
        # Update temperature
        self.actual_temp += heat_gain - heat_loss
        if self.actual_temp <= self.starting_temp:
            self.actual_temp = self.starting_temp
            
        
        # Log data
        self.time_history.append(self.env.now)
        self.temp_history.append(self.actual_temp)
        self.sensor_history.append(self.get_sensor_reading())
        self.heater_history.append(self.heater_on)
        self.heater_temp_history.append(heater_temp)
        
        # Checks if anomaly occured / is occuring 
        is_anomalous = 0 
        for anomaly_type in ['gradual', 'abrupt', 'sensor fail']:
            anomaly = self.check_anomaly(anomaly_type)
            if anomaly:
                is_anomalous = 1
                if anomaly.get('type') == 'abrupt_window':
                    is_anomalous = 1
                
        self.anomaly_history.append(is_anomalous)

class DigitalTwin:    
    def __init__(self, env, ideal_temp, threshold, starting_temp):
        self.env = env
        self.ideal_temp = ideal_temp
        self.threshold = threshold
        self.starting_temp = starting_temp
        
        # Simplified heat change model and starting states(Variables are assumed to be constant)
        self.predicted_temp = starting_temp  
        self.model_heater_rate = 0.5 
        self.model_heat_loss_coefficient = 0.02  
        self.heater_current_state = False
        
        # P-feedback correction  
        self.correction_factor = 0.01 
        
        # Data logging
        self.predicted_temp_history = []
        self.time_history = []
        
    def update_prediction(self, sensor_reading=None, heater_state=False):
        # Predict temp change using DT internal model
        if heater_state:
            heat_gain = self.model_heater_rate
        else:
            heat_gain = 0
            
        # Heat loss (constant rate)
        heat_loss = self.model_heat_loss_coefficient 
        
        # Update prediction
        self.predicted_temp += (heat_gain - heat_loss)
        
        # Apply weak correction based on sensor reading
        if sensor_reading is not None:
            error = sensor_reading - self.predicted_temp
            correction = error * self.correction_factor
            self.predicted_temp += correction
            
        # Log data
        self.time_history.append(self.env.now)
        self.predicted_temp_history.append(self.predicted_temp)
        
    def heater_control(self):
       # Sends command to turn on heater if below threshold
        if self.predicted_temp < self.ideal_temp - self.threshold:
            return True  # Turn heater on
        elif self.predicted_temp > self.ideal_temp + self.threshold:
            return False  # Turn heater off
        else:
            return self.heater_current_state # Maintain current state

class IncubatorSimulation:
    
    def __init__(self, duration, ideal_temp, threshold, starting_temp, configuration):
        self.env = simpy.Environment()
        self.duration = duration
        self.threshold = threshold
        self.config = configuration
        
        # Create incubator and digital twin class
        self.incubator = Incubator(self.env, starting_temp, self.duration, self.config)
        self.digital_twin = DigitalTwin(self.env, ideal_temp=ideal_temp, threshold=threshold, starting_temp=starting_temp)
        
        # Start simulation processes
        self.env.process(self.run_incubator())
        self.env.process(self.run_dt())
        
        # Logs error
        self.error = []
        
        
        
    def run_incubator(self):
        # Updates simulator 
        while self.env.now < self.duration:
            self.incubator.update_temperature()
            yield self.env.timeout(1) 
            
    def run_dt(self):
        # Updates Dt 
        while self.env.now < self.duration:
            # Gets sensor reading
            sensor_reading = self.incubator.get_sensor_reading()
            
            # Update prediction
            self.digital_twin.update_prediction(sensor_reading=sensor_reading, heater_state=self.digital_twin.heater_current_state)
            
            # Determine if current temp within threshold
            heater_command = self.digital_twin.heater_control()
            
            # Send command to incubator heater if outside of threshold
            if heater_command != self.digital_twin.heater_current_state:
                latency = random.randint(1, 10)  # 1-10 second latency
                self.incubator.set_heater(heater_command, latency)
                self.digital_twin.heater_current_state = heater_command
                
            yield self.env.timeout(1)  # Control loop runs every second
            
    def run(self):
        # Runs the simulation 
        self.env.run(self.duration)
        
    def plot_results(self):
        # Plots graphs 
        fig,(ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Temp plot
        ax1.plot(self.incubator.time_history, self.incubator.temp_history, color = 'b', label='Actual Temperature', linewidth=2)
        ax1.plot(self.digital_twin.time_history, self.digital_twin.predicted_temp_history, color='r', label='Predicted Temperature', linewidth=2)
        ax1.axhline(y=self.digital_twin.ideal_temp, color='g', linestyle=':',label=f'Ideal Temperature ({self.digital_twin.ideal_temp}°C)')
        
        # Doesn't label every plot only first plot
        gradual_plotted = False
        abrupt_plotted = False
        sensor_plotted = False
        
        for anomaly in self.incubator.anomalies:
            if anomaly['type'] == 'gradual':
                if not gradual_plotted:
                    ax1.axvspan(anomaly['start'], anomaly['start'] + anomaly['duration'], alpha=0.2, color='orange', label='Gradual Environment Drift')
                    gradual_plotted = True
                else: 
                    ax1.axvspan(anomaly['start'], anomaly['start'] + anomaly['duration'], alpha=0.2, color='orange')
            elif anomaly['type'] == 'abrupt':
                if not abrupt_plotted:
                    ax1.axvline(x=anomaly['start'], alpha=0.8, color='red', linestyle='--', label='Sensor Interference Anomaly')
                    abrupt_plotted = True
                else:
                    ax1.axvline(x=anomaly['start'], alpha=0.8, color='red', linestyle='--')
            if anomaly['type'] == 'sensor fail':
                if not sensor_plotted:
                    ax1.axvspan(anomaly['start'], anomaly['start'] + anomaly['duration'], alpha=0.2, color='orange', label='Sensor Failure')
                    sensor_plotted = True
                else: 
                    ax1.axvspan(anomaly['start'], anomaly['start'] + anomaly['duration'], alpha=0.2, color='orange')
        
        ax1.fill_between(self.incubator.time_history, (self.digital_twin.ideal_temp - self.digital_twin.threshold), (self.digital_twin.ideal_temp + self.digital_twin.threshold), alpha=0.2, color='green', label='Threshold')
        
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Temperature: Actual v Predicted')
        
        # Heater state plot
        heater_binary = [1 if h else 0 for h in self.incubator.heater_history]
        ax2.step(self.incubator.time_history, heater_binary, 'k-', where='post', linewidth=2)
        ax2.fill_between(self.incubator.time_history, heater_binary, step='post', alpha=0.3)
        ax2.set_ylabel('Heater State')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['OFF', 'ON'])
        ax2.grid(True)
        ax2.set_title('Heater Control')
        
        plt.tight_layout()
        plt.show()


# In[203]:


##### Runs simulator ######
DURATION = 2000 # seconds
IDEAL_TEMP = 80 # Celsius 
THRES = 5 # Threshold around the ideal temp
STARTING_TEMP = 25 # Starting temp

## Configurations: 0 = normal, 1 = gradual drifts, 2 = abrupt drifts, 3 = sensor failure, 4 = Mixed gradual & Abrupt 
## Config 4 should be used with very long durations due to increased chance rates 

CONFIG = 4

if __name__ == "__main__":
    sim = IncubatorSimulation(duration=DURATION, ideal_temp=IDEAL_TEMP, threshold =THRES, starting_temp = STARTING_TEMP, configuration = CONFIG) 
    sim.run()
    
    # Plot results
    sim.plot_results()
    
    # Logged activity
    actual_temps = np.array(sim.incubator.temp_history)
    predicted_temps = np.array(sim.digital_twin.predicted_temp_history)
    heater_activity = np.array(sim.incubator.heater_history)
    time_step = np.array(sim.digital_twin.time_history)
    temp_error = np.array(actual_temps - predicted_temps)
    temp_error = abs(temp_error)
    error_rate_change = np.diff(temp_error, prepend=temp_error[0]) # Change in error from last time step
    anomaly = np.array(sim.incubator.anomaly_history)


# In[180]:


data = np.array([temp_error, heater_activity, error_rate_change, anomaly]).T
df = pd.DataFrame(data, columns=['temp', 'heat', 'rate','anom'])
#df.to_csv('dataset/....',header=False)

