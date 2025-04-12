import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class ResourceAllocationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("5G Resource Allocation Predictor")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f4f8")

        # Load model with custom objects
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
        self.style.configure("TLabel", font=("Segoe UI", 12), background="#f0f4f8")
        self.style.configure("TButton", font=("Segoe UI", 12, "bold"))
        self.style.configure("TEntry", font=("Segoe UI", 12))
        self.style.theme_use("clam")

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.configure(style="Main.TFrame")
        self.style.configure("Main.TFrame", background="#f0f4f8")

        # Title
        self.title_label = ttk.Label(
            self.main_frame,
            text="5G Resource Allocation Predictor",
            font=("Segoe UI", 18, "bold"),
            foreground="#2c3e50"
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Input frame
        self.input_frame = ttk.LabelFrame(
            self.main_frame,
            text="Input Parameters",
            padding=10,
            style="Input.TLabelframe"
        )
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=10)
        self.style.configure("Input.TLabelframe", font=("Segoe UI", 12))

        # Input fields
        self.labels = ["Bandwidth (Mbps)", "Latency (ms)", "Power (mW)", "SNR (dB)", "Interference (dB)"]
        self.entries = {}
        for i, label in enumerate(self.labels):
            ttk.Label(self.input_frame, text=label).grid(row=i, column=0, sticky="w", pady=5)
            entry = ttk.Entry(self.input_frame, width=20)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries[label] = entry

        # Button frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=2, column=0, pady=20)

        self.predict_button = ttk.Button(
            self.button_frame,
            text="Predict",
            command=self.predict,
            style="Predict.TButton"
        )
        self.predict_button.grid(row=0, column=0, padx=5)
        self.style.configure("Predict.TButton", background="#3498db", foreground="white")

        self.clear_button = ttk.Button(
            self.button_frame,
            text="Clear",
            command=self.clear_inputs,
            style="Clear.TButton"
        )
        self.clear_button.grid(row=0, column=1, padx=5)
        self.style.configure("Clear.TButton", background="#e74c3c", foreground="white")

        # Result frame
        self.result_frame = ttk.LabelFrame(
            self.main_frame,
            text="Prediction Results",
            padding=10,
            style="Result.TLabelframe"
        )
        self.result_frame.grid(row=3, column=0, sticky="ew", padx=10)
        self.style.configure("Result.TLabelframe", font=("Segoe UI", 12))

        self.spectrum_label = ttk.Label(self.result_frame, text="Predicted Spectrum: -")
        self.spectrum_label.grid(row=0, column=0, sticky="w", pady=5)

        self.power_label = ttk.Label(self.result_frame, text="Allocated Power: -")
        self.power_label.grid(row=1, column=0, sticky="w", pady=5)

        # Plot frame
        self.plot_frame = ttk.LabelFrame(
            self.main_frame,
            text="Spectrum Allocation Plot",
            padding=10
        )
        self.plot_frame.grid(row=1, column=1, rowspan=3, sticky="ns", padx=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack()

        # Status label
        self.status_label = ttk.Label(
            self.main_frame,
            text="Ready",
            font=("Segoe UI", 10),
            foreground="#7f8c8d"
        )
        self.status_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

    def validate_inputs(self):
        try:
            inputs = []
            for label in self.labels:
                value = self.entries[label].get().strip()
                if not value:
                    raise ValueError(f"{label} is empty")
                value = float(value)
                if value <= 0:
                    raise ValueError(f"{label} must be positive")
                inputs.append(value)
            return np.array(inputs).reshape(1, -1)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return None

    def predict(self):
        inputs = self.validate_inputs()
        if inputs is None:
            self.status_label.config(text="Prediction failed", foreground="#e74c3c")
            return

        try:
            scaled_inputs = self.scaler.transform(inputs)
            prediction = self.model.predict(scaled_inputs)
            spectrum, power = prediction[0]

            self.spectrum_label.config(text=f"Predicted Spectrum: {spectrum:.2f} MHz")
            self.power_label.config(text=f"Allocated Power: {power:.2f} mW")
            self.status_label.config(text="Prediction successful", foreground="#2ecc71")

            self.ax.clear()
            self.ax.bar(["Spectrum", "Power"], [spectrum, power], color=["#3498db", "#e74c3c"])
            self.ax.set_ylabel("Value")
            self.ax.set_title("Prediction Results")
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {e}")
            self.status_label.config(text="Prediction failed", foreground="#e74c3c")

    def clear_inputs(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.spectrum_label.config(text="Predicted Spectrum: -")
        self.power_label.config(text="Allocated Power: -")
        self.ax.clear()
        self.canvas.draw()
        self.status_label.config(text="Inputs cleared", foreground="#7f8c8d")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResourceAllocationGUI(root)
    root.mainloop()