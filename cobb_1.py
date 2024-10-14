import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tensorflow.keras.models import load_model

# Parameters for data simulation
mean = 50
std_dev = 10
seasonal_amplitude = 5
seasonal_period = 20
noise_factor = 3

# Parameters for anomaly detection
threshold = 8.684171334607531e-06  # Adjusted MSE threshold for single data point
# Ensure this threshold is appropriate for point-wise MSE

def simulate_data_stream():
    """Simulates a continuous data stream with seasonal patterns and noise."""
    t = 0
    while True:
        # Generate a seasonal pattern with random noise
        seasonal_pattern = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
        noise = random.uniform(-noise_factor, noise_factor)
        data_point = np.random.normal(loc=mean + seasonal_pattern, scale=std_dev) + noise

        yield data_point
        t += 1  # Increment time for the seasonal function

def detect_anomalies_autoencoder_point(data_stream, autoencoder, threshold):
    """Detects anomalies in the data stream using a pre-trained autoencoder on individual points."""
    time_step = 0  # To keep track of the time for plotting

    for data_point in data_stream:
        time_step += 1

        # Prepare the data for the autoencoder
        # Assuming the autoencoder expects input shape (batch_size, 1, 1)
        # Adjust reshape according to your model's expected input shape
        input_data = np.array(data_point).reshape(1, 1, 1)

        # Reconstruct the data using the autoencoder
        reconstructed = autoencoder.predict(input_data)
        reconstructed = reconstructed.reshape(1)  # Reshape to match original data point

        # Calculate Mean Squared Error between original and reconstructed data point
        mse = np.mean(np.power(data_point - reconstructed[0], 2))

        # Detect anomaly based on the threshold
        if mse > threshold:
            # Anomaly detected: Flag the data point
            print(f"Anomaly detected at time {time_step}: {data_point:.2f} (MSE: {mse:.6f})")
            plt.scatter(time_step, data_point, color='red')  # Plotting anomaly
        else:
            # Normal data point
            plt.scatter(time_step, data_point, color='blue')  # Plotting normal data

        # Optional: Plot the MSE for visualization
        # plt.scatter(time_step, mse, color='green', marker='x')  # Uncomment to plot MSE

        # Real-time plot update
        plt.pause(0.001)  # Pause briefly to update the plot

def main():
    # Load the pre-trained autoencoder model
    try:
        autoencoder = load_model('autoencoder_model.keras')
        print("Autoencoder model loaded successfully.")
    except Exception as e:
        print(f"Error loading the autoencoder model: {e}")
        return

    # Set up the plot
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(15, 7))
    plt.title('Real-Time Data Stream with Anomaly Detection using Autoencoder')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Initialize the data stream simulation and anomaly detection
    data_stream = simulate_data_stream()
    detect_anomalies_autoencoder_point(data_stream, autoencoder, threshold)

if __name__ == "__main__":
    main()
