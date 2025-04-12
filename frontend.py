import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import seaborn as sns
import pandas as pd  # Added to fix NameError
from datetime import datetime

class HighTechResourceAllocationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("5G Resource Allocation Hub")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")

        # Load model and scaler
        try:
            self.model = tf.keras.models.load_model(
                "saved_models/5g_model.h5",
                custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
            )
            with open("saved_models/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model/scaler: {e}")
            self.root.quit()

        # Style configuration
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Orbitron", 12), foreground="#e94560", background="#1a1a2e")
        self.style.configure("TButton", font=("Orbitron", 12, "bold"), background="#00d4ff", foreground="#1a1a2e")
        self.style.configure("TEntry", font=("Orbitron", 12), fieldbackground="#2e2e48")
        self.style.configure("TFrame", background="#1a1a2e")
        self.style.configure("Heading.TLabel", font=("Orbitron", 16, "bold"), foreground="#00ffcc")
        self.style.theme_use("clam")

        # Main frame with gradient effect
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Header
        self.header_label = ttk.Label(
            self.main_frame,
            text="5G Resource Allocation Hub",
            style="Heading.TLabel"
        )
        self.header_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Input frame
        self.input_frame = ttk.LabelFrame(
            self.main_frame,
            text="Input Parameters",
            padding=10,
            labelanchor="n",
            style="TFrame"
        )
        self.input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.input_frame.columnconfigure(1, weight=1)

        self.labels = ["Bandwidth (Mbps)", "Latency (ms)", "Power (mW)", "SNR (dB)", "Interference (dB)"]
        self.entries = {}
        for i, label in enumerate(self.labels):
            ttk.Label(self.input_frame, text=label).grid(row=i, column=0, sticky="w", pady=5)
            entry = ttk.Entry(self.input_frame, width=15)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
            self.entries[label] = entry

        # Control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=2, column=0, pady=10)
        ttk.Button(self.control_frame, text="Predict", command=self.predict, style="TButton").grid(row=0, column=0, padx=5)
        ttk.Button(self.control_frame, text="Clear", command=self.clear_inputs, style="TButton").grid(row=0, column=1, padx=5)
        ttk.Button(self.control_frame, text="Reset Graphs", command=self.reset_graphs, style="TButton").grid(row=0, column=2, padx=5)
        self.detail_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Detailed Mode", variable=self.detail_var, command=self.toggle_details, style="TCheckbutton").grid(row=0, column=3, padx=5)

        # Graph frames
        self.graph_frame1 = ttk.LabelFrame(self.main_frame, text="Prediction Trend", padding=5, style="TFrame")
        self.graph_frame1.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.fig1, self.ax1 = plt.subplots(figsize=(5, 4))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.graph_frame1)
        self.canvas1.get_tk_widget().pack()
        NavigationToolbar2Tk(self.canvas1, self.graph_frame1).update()

        self.graph_frame2 = ttk.LabelFrame(self.main_frame, text="Input-Output Scatter", padding=5, style="TFrame")
        self.graph_frame2.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
        self.fig2, self.ax2 = plt.subplots(figsize=(5, 4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.graph_frame2)
        self.canvas2.get_tk_widget().pack()
        NavigationToolbar2Tk(self.canvas2, self.graph_frame2).update()

        self.graph_frame3 = ttk.LabelFrame(self.main_frame, text="Correlation Heatmap", padding=5, style="TFrame")
        self.graph_frame3.grid(row=1, column=2, rowspan=3, padx=10, pady=10, sticky="nsew")
        self.fig3, self.ax3 = plt.subplots(figsize=(6, 5))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.graph_frame3)
        self.canvas3.get_tk_widget().pack()
        NavigationToolbar2Tk(self.canvas3, self.graph_frame3).update()

        # Result frame
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Results & Metrics", padding=10, style="TFrame")
        self.result_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.result_labels = {
            "Spectrum": ttk.Label(self.result_frame, text="Predicted Spectrum: - MHz"),
            "Power": ttk.Label(self.result_frame, text="Allocated Power: - mW"),
            "Loss": ttk.Label(self.result_frame, text="Test Loss: -"),
            "MSE": ttk.Label(self.result_frame, text="Test MSE: -")
        }
        for i, (key, label) in enumerate(self.result_labels.items()):
            label.grid(row=i, column=0, sticky="w", pady=2)

        # Status and log frame
        self.status_frame = ttk.LabelFrame(self.main_frame, text="System Log", padding=5, style="TFrame")
        self.status_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.status_text = tk.Text(self.status_frame, height=5, width=80, bg="#2e2e48", fg="#00ffcc", font=("Orbitron", 10))
        self.status_text.grid(row=0, column=0, sticky="nsew")
        self.status_frame.columnconfigure(0, weight=1)

        # Load historical data for graphs
        self.historical_data = pd.read_csv("datasets/5g_data.csv") if os.path.exists("datasets/5g_data.csv") else None
        self.plot_initial_data()

        # Configure weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.columnconfigure(2, weight=2)
        for i in range(6):
            self.main_frame.rowconfigure(i, weight=1)

    def validate_inputs(self):
        try:
            inputs = []
            ranges = {"Bandwidth (Mbps)": (1, 100), "Latency (ms)": (1, 500), "Power (mW)": (1, 200), "SNR (dB)": (0, 100), "Interference (dB)": (0, 50)}
            for label in self.labels:
                value = self.entries[label].get().strip()
                if not value:
                    raise ValueError(f"{label} is empty")
                value = float(value)
                min_val, max_val = ranges[label]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{label} must be between {min_val} and {max_val}")
                inputs.append(value)
            return np.array(inputs).reshape(1, -1)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return None

    def predict(self):
        inputs = self.validate_inputs()
        if inputs is None:
            self.log_status("Prediction failed: Invalid inputs", "red")
            return

        try:
            scaled_inputs = self.scaler.transform(inputs)
            prediction = self.model.predict(scaled_inputs)
            spectrum, power = prediction[0]
            loss, mse = self.model.evaluate(scaled_inputs, np.array([[spectrum, power]]), verbose=0)

            self.result_labels["Spectrum"].config(text=f"Predicted Spectrum: {spectrum:.2f} MHz")
            self.result_labels["Power"].config(text=f"Allocated Power: {power:.2f} mW")
            self.result_labels["Loss"].config(text=f"Test Loss: {loss:.2f}")
            self.result_labels["MSE"].config(text=f"Test MSE: {mse:.2f}")
            self.log_status("Prediction successful", "green")

            self.update_graphs(spectrum, power, inputs[0][0])  # Update with bandwidth as x-axis example

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {e}")
            self.log_status(f"Prediction failed: {e}", "red")

    def update_graphs(self, spectrum, power, bandwidth):
        # Trend Line
        self.ax1.clear()
        self.ax1.plot([0, 1], [0, spectrum], label="Spectrum Trend", color="#00d4ff")
        self.ax1.plot([0, 1], [0, power], label="Power Trend", color="#e94560")
        self.ax1.set_title("Prediction Trend Over Time", color="#00ffcc")
        self.ax1.set_xlabel("Time Step", color="#00ffcc")
        self.ax1.set_ylabel("Value", color="#00ffcc")
        self.ax1.legend()
        self.ax1.tick_params(colors="#00ffcc")
        self.canvas1.draw()

        # Scatter Plot
        self.ax2.clear()
        self.ax2.scatter([bandwidth], [spectrum], color="#00d4ff", label="Spectrum")
        self.ax2.scatter([bandwidth], [power], color="#e94560", label="Power")
        self.ax2.set_title("Input vs Output", color="#00ffcc")
        self.ax2.set_xlabel("Bandwidth (Mbps)", color="#00ffcc")
        self.ax2.set_ylabel("Predicted Value", color="#00ffcc")
        self.ax2.legend()
        self.ax2.tick_params(colors="#00ffcc")
        self.canvas2.draw()

        # Correlation Heatmap (static for now)
        if self.historical_data is not None:
            self.ax3.clear()
            correlation_matrix = self.historical_data[["bandwidth", "latency", "power", "snr", "interference", "spectrum", "allocated_power"]].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="cool", linewidths=0.5, ax=self.ax3)
            self.ax3.set_title("Feature Correlation", color="#00ffcc")
            self.ax3.tick_params(colors="#00ffcc")
            self.canvas3.draw()

    def plot_initial_data(self):
        if self.historical_data is not None:
            self.update_graphs(0, 0, 0)  # Initial plot with zeros

    def reset_graphs(self):
        self.plot_initial_data()
        self.log_status("Graphs reset", "yellow")

    def toggle_details(self):
        if self.detail_var.get():
            self.result_frame.config(text="Results & Detailed Metrics")
            self.result_labels["Loss"].grid()
            self.result_labels["MSE"].grid()
        else:
            self.result_frame.config(text="Results & Metrics")
            self.result_labels["Loss"].grid_remove()
            self.result_labels["MSE"].grid_remove()
        self.log_status(f"Detailed Mode: {'On' if self.detail_var.get() else 'Off'}", "yellow")

    def clear_inputs(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        for label in self.result_labels.values():
            label.config(text=f"{label.cget('text').split(':')[0]}: -")
        self.reset_graphs()
        self.log_status("Inputs cleared", "yellow")

    def log_status(self, message, color):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.tag_config("red", foreground="red")
        self.status_text.tag_config("green", foreground="green")
        self.status_text.tag_config("yellow", foreground="yellow")
        self.status_text.see(tk.END)
        self.status_text.tag_add(color, f"{len(self.status_text.get('1.0', tk.END).splitlines())-1}.0", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = HighTechResourceAllocationGUI(root)
    root.mainloop()