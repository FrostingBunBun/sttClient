# ui.py
import tkinter as tk
from tkinter import simpledialog, ttk
import sounddevice as sd

class STTClientUI:
    def __init__(self, root, config, on_send_text, on_try_connect_again, on_change_keybind, on_toggle_stt):
        self.root = root
        self.root.title("Live STT WebSocket Client")
        self.root.geometry("600x500")

        self.config = config
        self.on_send_text = on_send_text
        self.on_try_connect_again = on_try_connect_again
        self.on_change_keybind = on_change_keybind
        self.on_toggle_stt = on_toggle_stt

        # UI Elements
        self.create_ui()

    def create_ui(self):
        """Create and organize the UI elements."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Text input field
        self.text_input_label = ttk.Label(self.main_frame, text="Enter Text (or use mic):")
        self.text_input_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.text_input = ttk.Entry(self.main_frame, width=50)
        self.text_input.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # Send button
        self.send_button = ttk.Button(self.main_frame, text="Send", command=self.on_send_text)
        self.send_button.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        # Try to Connect Again button
        self.connect_button = ttk.Button(self.main_frame, text="Try to Connect Again", command=self.on_try_connect_again)
        self.connect_button.grid(row=3, column=0, columnspan=2, pady=(0, 10))

        # Mic Selection Dropdown
        self.mic_label = ttk.Label(self.main_frame, text="Select Microphone:")
        self.mic_label.grid(row=4, column=0, sticky=tk.W, pady=(0, 5))

        self.mic_list = [f"{i}: {device['name']}" for i, device in enumerate(sd.query_devices())]
        self.selected_mic = tk.StringVar()
        self.selected_mic.set(self.mic_list[self.config.get("mic_index", 0)])  # Default to saved mic

        self.mic_dropdown = ttk.Combobox(self.main_frame, textvariable=self.selected_mic, values=self.mic_list, state="readonly")
        self.mic_dropdown.grid(row=5, column=0, columnspan=2, pady=(0, 10))

        # Keybind Selection
        self.keybind_label = ttk.Label(self.main_frame, text=f"Current Keybind: {self.config.get('keybind', 'space')}")
        self.keybind_label.grid(row=6, column=0, sticky=tk.W, pady=(0, 5))

        self.keybind_button = ttk.Button(self.main_frame, text="Change Keybind", command=self.on_change_keybind)
        self.keybind_button.grid(row=7, column=0, columnspan=2, pady=(0, 10))

        # STT Toggle Button
        self.listening = False
        self.stt_button = ttk.Button(self.main_frame, text="Start STT", command=self.on_toggle_stt)
        self.stt_button.grid(row=8, column=0, columnspan=2, pady=(0, 10))

        # Log area
        self.log_area = tk.Text(self.main_frame, height=10, width=60, state=tk.DISABLED)
        self.log_area.grid(row=9, column=0, columnspan=2, pady=(0, 10))

        # Scrollbar for log area
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.log_area.yview)
        self.log_area.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.grid(row=9, column=2, sticky=tk.NS)

    def log(self, message):
        """Log messages in UI."""
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state=tk.DISABLED)
        self.log_area.yview(tk.END)