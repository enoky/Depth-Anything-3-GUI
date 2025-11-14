import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import threading
import queue
import torch
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt  # For the _test function
from depth_anything_3.api import DepthAnything3

# --- Content from utils.py ---

# Configure a logger for this module
_logger = logging.getLogger(__name__)
# Basic logging config for the script, in case it's not run as a module
logging.basicConfig(level=logging.INFO)

def normalize_video_data(
    video_data: np.ndarray,
    use_percentile_norm: bool,
    low_perc: float,
    high_perc: float
) -> np.ndarray:
    """Normalizes video data to the 0-1 range."""
    if video_data is None or video_data.size == 0:
        _logger.warning("Cannot normalize empty or invalid video data.")
        return np.array([]) # Return empty array

    _logger.debug(f"Normalizing video data. Shape: {video_data.shape}")
    
    normalized_video = video_data.copy().astype(np.float32)
    min_val_for_norm, max_val_for_norm = np.min(normalized_video), np.max(normalized_video)
    method_str = "absolute"

    if use_percentile_norm:
        # Check for sufficient data points for a meaningful percentile
        if normalized_video.ndim > 0 and normalized_video.size > 20:
            try:
                min_val_for_norm = np.percentile(normalized_video, low_perc)
                max_val_for_norm = np.percentile(normalized_video, high_perc)
                method_str = f"percentile ({low_perc}%-{high_perc}%)"
            except IndexError: # Fallback for very small arrays where percentile might fail
                _logger.debug("Percentile calculation failed, falling back to absolute min/max.")
                method_str = "absolute (percentile fallback)"
        else:
            _logger.debug(f"Array too small for robust percentile, using absolute min/max.")
            method_str = "absolute (percentile fallback)"

    _logger.debug(f"Normalizing with method: {method_str}. Range: {min_val_for_norm:.4f} to {max_val_for_norm:.4f}")

    if abs(max_val_for_norm - min_val_for_norm) < 1e-6:
        _logger.warning("Normalization range is very small; video may appear flat.")
        # Set to mid-gray, but if original was already normalized, respect that
        flat_value = 0.5 if not (0.0 <= min_val_for_norm <= 1.0) else np.clip(min_val_for_norm, 0.0, 1.0)
        normalized_video = np.full_like(normalized_video, flat_value, dtype=np.float32)
    else:
        normalized_video = (normalized_video - min_val_for_norm) / (max_val_for_norm - min_val_for_norm)
    
    normalized_video = np.clip(normalized_video, 0.0, 1.0)
    _logger.debug(f"Normalization complete. Final min/max: {np.min(normalized_video):.4f}, {np.max(normalized_video):.4f}")
    return normalized_video

def apply_gamma_correction_to_video(
    video_data: np.ndarray,
    gamma_value: float
) -> np.ndarray:
    """Applies gamma correction to video data."""
    if video_data is None or video_data.size == 0:
        return np.array([])

    processed_video = video_data.copy()
    actual_gamma = max(0.1, gamma_value) # Prevent gamma of 0

    if abs(actual_gamma - 1.0) > 1e-3:
        _logger.debug(f"Applying Gamma ({actual_gamma:.2f}) to video.")
        # Ensure data is clipped to a valid range for power function
        processed_video = np.power(np.clip(processed_video, 0, 1), 1.0 / actual_gamma)
        processed_video = np.clip(processed_video, 0, 1) # Re-clip to handle potential floating point inaccuracies
    else:
        _logger.debug(f"Gamma value is effectively 1.0; no transform applied.")
    return processed_video

# --- End of content from utils.py ---


# --- Content from depth_scaler.py ---

def minmax_normalize(frame, min_value, max_value):
    if torch.is_tensor(min_value):
        min_value = min_value.to(frame.device)
        max_value = max_value.to(frame.device)
    frame = 1.0 - ((frame - min_value) / (max_value - min_value))
    frame = frame.clamp(0.0, 1.0)
    frame = frame.nan_to_num()
    return frame


class MinMaxBuffer():
    def __init__(self, size, dtype, device):
        assert size > 0
        self.count = 0
        self.size = size * 2
        self.data = torch.zeros(self.size, dtype=dtype).to(device)

    def _add(self, value):
        index = self.count % self.size
        self.data[index] = value
        self.count += 1

    def _fill(self, min_value, max_value):
        self.data[0::2] = min_value
        self.data[1::2] = max_value

    def add(self, min_value, max_value):
        if self.count == 0:
            self._fill(min_value, max_value)
            self.count = 2
        else:
            self._add(min_value)
            self._add(max_value)

    def is_filled(self):
        return self.count >= self.size

    def get_minmax(self):
        return self.data.amin(), self.data.amax()


class EMAMinMaxScaler():
    #   SimpleMinMaxScaler: decay=0, buffer_size=1
    # IncrementalEMAScaler: decay=0.75, buffer_size=1
    #      WindowEMAScaler: decay=0.9, buffer_size=30
    def __init__(self, decay=0, buffer_size=1):
        self.frame_queue = []
        assert buffer_size > 0
        self.reset(decay=decay, buffer_size=buffer_size)

    def reset(self, decay=None, buffer_size=None, **kwargs):
        # assert len(self.frame_queue) == 0  # need flush

        if decay is not None:
            self.decay = float(decay)
        if buffer_size is not None:
            self.buffer_size = int(buffer_size)
        self.min_value = None
        self.max_value = None
        self.frame_queue = []
        self.minmax_buffer = None

    def get_minmax(self):
        assert self.minmax_buffer is not None and self.minmax_buffer.is_filled()
        return self.minmax_buffer.get_minmax()

    def __call__(self, frame, return_minmax=False):
        return self.update(frame, return_minmax=return_minmax)

    def update(self, frame, return_minmax=False):
        if self.minmax_buffer is None:
            self.minmax_buffer = MinMaxBuffer(self.buffer_size, dtype=frame.dtype, device=frame.device)
        self.frame_queue.append(frame)
        self.minmax_buffer.add(frame.amin(), frame.amax())
        if not self.minmax_buffer.is_filled():
            # queued
            if return_minmax:
                return None, None, None
            else:
                return None

        min_value, max_value = self.get_minmax()
        if self.min_value is None:
            self.min_value = min_value
            self.max_value = max_value
        else:
            self.min_value = self.decay * self.min_value + (1. - self.decay) * min_value
            self.max_value = self.decay * self.max_value + (1. - self.decay) * max_value

        frame = self.frame_queue.pop(0)
        frame = minmax_normalize(frame, self.min_value, self.max_value)

        if return_minmax:
            return (frame, self.min_value, self.max_value)
        else:
            return frame

    def flush(self, return_minmax=False):
        if not self.frame_queue:
            self.reset()
            return []

        if self.min_value is None:
            min_value, max_value = self.minmax_buffer.get_minmax()
        else:
            min_value, max_value = self.min_value, self.max_value

        if return_minmax:
            frames = [(minmax_normalize(frame, min_value, max_value),
                       min_value, max_value)
                      for frame in self.frame_queue]
            self.reset()
            return frames
        else:
            frames = [minmax_normalize(frame, min_value, max_value)
                      for frame in self.frame_queue]
            self.reset()
            return frames

def _test():
    x = [float(i) for i in range(100)]
    zeros = [0 for i in range(100)]

    x = torch.tensor(zeros + x + list(reversed(x)) + zeros, dtype=torch.float32)
    x = torch.stack([x, x + 10]).permute(1, 0).contiguous()

    scaler = EMAMinMaxScaler(decay=0.9, buffer_size=22)
    min_values = []
    max_values = []
    for frame in x:
        frame, min_value, max_value = scaler.update(frame, return_minmax=True)
        if min_value is not None:
            min_values.append(min_value)
            max_values.append(max_value)
    for frame, min_value, max_value in scaler.flush(return_minmax=True):
        min_values.append(min_value)
        max_values.append(max_value)

    min_values = torch.tensor(min_values)
    max_values = torch.tensor(max_values)

    x = torch.stack([
        x.permute(1, 0)[0],
        x.permute(1, 0)[1],
        min_values,
        max_values,
    ]).permute(1, 0)
    plt.plot(x)
    plt.show()

# --- End of content from depth_scaler.py ---


# --- Content from create_depth_video.py ---

def process_video(video_input, video_output, model, process_res, batch_size=1, progress_callback=None, stop_event=None, scaler=None,
                  use_gamma=False, gamma_value=1.0):
    """
    Processes a single video file to create a depth map video.
    The model is passed as an argument to avoid reloading it for each video.
    Includes post-processing options.
    """
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_input}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height), isColor=True)

    frames_buffer = []
    
    def write_frames(frames_np):
        for frame in frames_np:
            # Post-processing is now applied before this function is called
            
            # OLD LINE (truncates, which nullifies dither):
            # grayscale_depth = (frame * 255).astype(np.uint8)
            
            # NEW LINE (rounds, which allows dither to work):
            grayscale_depth = np.round(frame * 255).astype(np.uint8)
            
            output_frame = cv2.cvtColor(grayscale_depth, cv2.COLOR_GRAY2BGR)
            output_frame = cv2.resize(output_frame, (frame_width, frame_height))
            out.write(output_frame)

    try:
        while True:
            if stop_event and stop_event.is_set():
                print("Processing stopped.")
                break
            
            ret, frame = cap.read()
            if ret:
                frames_buffer.append(frame)

            if len(frames_buffer) == batch_size or (not ret and len(frames_buffer) > 0):
                frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer]

                prediction = model.inference(
                    frames_rgb,
                    process_res=process_res
                )
                
                processed_frames_to_write = []
                for i in range(len(frames_buffer)):
                    depth_map = prediction.depth[i]
                    
                    if scaler:
                        frame_tensor = scaler(torch.from_numpy(depth_map).to(model.device))
                        if frame_tensor is not None:
                            # Convert tensor to numpy for post-processing
                            frame_np = frame_tensor.cpu().numpy()
                            
                            # Apply post-processing
                            if use_gamma:
                                frame_np = apply_gamma_correction_to_video(frame_np, gamma_value)
                                
                            processed_frames_to_write.append(frame_np)
                    else:
                        # Basic normalization if no scaler
                        depth_tensor = torch.from_numpy(depth_map)
                        depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
                        frame_np = depth_tensor.cpu().numpy()
                        processed_frames_to_write.append(frame_np)

                if processed_frames_to_write:
                    write_frames(processed_frames_to_write)

                if progress_callback:
                    progress_callback(len(frames_buffer))
                
                frames_buffer = []

            if not ret:
                break
    finally:
        if scaler:
            # Process and write any remaining frames from the scaler's buffer
            flushed_frames = scaler.flush()
            flushed_np = [f.cpu().numpy() for f in flushed_frames]
            
            final_frames_to_write = []
            for frame_np in flushed_np:
                if use_gamma:
                    frame_np = apply_gamma_correction_to_video(frame_np, gamma_value)
                final_frames_to_write.append(frame_np)
            
            if final_frames_to_write:
                write_frames(final_frames_to_write)

        cap.release()
        out.release()

# --- End of content from create_depth_video.py ---


# --- Content from gui.py ---

class DepthVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Video Creator")
        self.root.geometry("500x550")
        self.config_file = "gui_config.json"
        
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self.progress_queue = queue.Queue()

        self.model = None
        self.device = None

        # Post-processing variables
        self.use_gamma = tk.BooleanVar(value=False)
        self.gamma_value = tk.DoubleVar(value=0.7)

        self.setup_ui()
        self.load_settings()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_queue()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- Input/Output Folders ---
        path_frame = ttk.LabelFrame(main_frame, text="Folders", padding="10")
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        path_frame.columnconfigure(1, weight=1)

        ttk.Label(path_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W)
        self.input_folder = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.input_folder).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(path_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=(5,0))

        ttk.Label(path_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W)
        self.output_folder = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.output_folder).grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Button(path_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=(5,0))

        # --- Settings ---
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Processing Resolution:").grid(row=0, column=0, sticky=tk.W)
        self.resolution = tk.IntVar(value=1274)
        ttk.Spinbox(settings_frame, from_=504, to=1932, increment=14, textvariable=self.resolution, width=10).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(settings_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W)
        self.batch_size = tk.IntVar(value=1)
        ttk.Spinbox(settings_frame, from_=1, to=100, increment=1, textvariable=self.batch_size, width=10).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Decay:").grid(row=2, column=0, sticky=tk.W)
        self.decay = tk.DoubleVar(value=0.9)
        ttk.Spinbox(settings_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.decay, width=10).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(settings_frame, text="Buffer Size:").grid(row=3, column=0, sticky=tk.W)
        self.buffer_size = tk.IntVar(value=30)
        ttk.Spinbox(settings_frame, from_=1, to=100, increment=1, textvariable=self.buffer_size, width=10).grid(row=3, column=1, sticky=tk.W)

        # --- Post-processing ---
        post_proc_frame = ttk.LabelFrame(main_frame, text="Post-processing", padding="10")
        post_proc_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        post_proc_frame.columnconfigure(1, weight=1)

        # Gamma Correction
        gamma_check = ttk.Checkbutton(post_proc_frame, text="Apply Gamma Correction", variable=self.use_gamma, command=self._toggle_widget_state)
        gamma_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5,0))
        
        self.gamma_label = ttk.Label(post_proc_frame, text="Gamma:")
        self.gamma_label.grid(row=5, column=0, sticky=tk.W, padx=(20,0))
        self.gamma_spinbox = ttk.Spinbox(post_proc_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.gamma_value, width=8)
        self.gamma_spinbox.grid(row=5, column=1, sticky=tk.W)

        self._toggle_widget_state() # Set initial state

        # --- Progress and Status ---
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        progress_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        self.overall_progress = tk.DoubleVar()
        self.overall_progress_bar = ttk.Progressbar(progress_frame, variable=self.overall_progress, maximum=100)
        self.overall_progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.overall_status_label = ttk.Label(progress_frame, text="Ready")
        self.overall_status_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)

        self.file_progress = tk.DoubleVar()
        self.file_progress_bar = ttk.Progressbar(progress_frame, variable=self.file_progress, maximum=100)
        self.file_progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5,0))
        self.file_status_label = ttk.Label(progress_frame, text="")
        self.file_status_label.grid(row=3, column=0, columnspan=2, sticky=tk.W)

        # --- Buttons ---
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, pady=(10, 0))
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing_thread, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        for frame in [path_frame, settings_frame, post_proc_frame, progress_frame]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=2)

    def _toggle_widget_state(self):
        # Gamma controls
        gamma_enabled = self.use_gamma.get()
        self.gamma_label.config(state=tk.NORMAL if gamma_enabled else tk.DISABLED)
        self.gamma_spinbox.config(state=tk.NORMAL if gamma_enabled else tk.DISABLED)

    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def load_settings(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.input_folder.set(config.get("input_folder", ""))
                    self.output_folder.set(config.get("output_folder", ""))
                    self.resolution.set(config.get("resolution", 1274))
                    self.batch_size.set(config.get("batch_size", 1))
                    self.decay.set(config.get("decay", 0.9))
                    self.buffer_size.set(config.get("buffer_size", 30))
        except (json.JSONDecodeError, IOError) as e:
            messagebox.showerror("Error", f"Could not load settings: {e}")

    def save_settings(self):
        config = {
            "input_folder": self.input_folder.get(),
            "output_folder": self.output_folder.get(),
            "resolution": self.resolution.get(),
            "batch_size": self.batch_size.get(),
            "decay": self.decay.get(),
            "buffer_size": self.buffer_size.get()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            messagebox.showwarning("Warning", f"Could not save settings: {e}")

    def on_closing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            if messagebox.askyesno("Exit", "Processing is ongoing. Are you sure you want to exit? This may corrupt the current video file."):
                self.stop_processing_thread()
                self.save_settings()
                self.root.destroy()
        else:
            self.save_settings()
            self.root.destroy()

    def check_queue(self):
        try:
            message = self.progress_queue.get_nowait()
            if "overall_progress" in message:
                self.overall_progress.set(message["overall_progress"])
                self.overall_status_label.config(text=message["overall_status"])
            if "file_progress" in message:
                self.file_progress.set(message["file_progress"])
                self.file_status_label.config(text=message["file_status"])
            if "processing_done" in message:
                self.processing_finished()
            if "error" in message:
                messagebox.showerror("Processing Error", message["error"])
                self.processing_finished()

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def start_processing(self):
        input_dir = self.input_folder.get()
        output_dir = self.output_folder.get()
        if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select valid input and output folders.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stop_processing.clear()
        
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def stop_processing_thread(self):
        self.stop_processing.set()
        self.stop_button.config(state=tk.DISABLED)
        self.overall_status_label.config(text="Stopping...")

    def processing_finished(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if not self.stop_processing.is_set():
            self.overall_status_label.config(text="All videos processed successfully!")
            self.overall_progress.set(100)
        else:
            self.overall_status_label.config(text="Processing stopped by user.")
        
        self.file_status_label.config(text="")
        self.file_progress.set(0)
        self.model = None # Release model from memory
        torch.cuda.empty_cache()


    def _processing_loop(self):
        try:
            input_dir = self.input_folder.get()
            output_dir = self.output_folder.get()
            resolution = self.resolution.get()
            batch_size = self.batch_size.get()
            decay = self.decay.get()
            buffer_size = self.buffer_size.get()
            
            video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if not video_files:
                self.progress_queue.put({"error": "No video files found in the input folder."})
                return

            # --- Load Model ---
            self.progress_queue.put({"overall_status": "Loading model...", "overall_progress": 0})
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE").to(self.device).eval()
            
            total_videos = len(video_files)
            for i, filename in enumerate(video_files):
                if self.stop_processing.is_set():
                    break
                
                input_path = os.path.join(input_dir, filename)
                output_filename = f"{os.path.splitext(filename)[0]}_depth.mp4"
                output_path = os.path.join(output_dir, output_filename)

                # --- Update Overall Progress ---
                overall_prog = (i / total_videos) * 100
                self.progress_queue.put({
                    "overall_progress": overall_prog,
                    "overall_status": f"Processing video {i+1} of {total_videos}: {filename}"
                })

                # --- Define Progress Callback for current file ---
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    self.progress_queue.put({"error": f"Could not open video file {filename}"})
                    continue
                total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                frames_done_for_file = 0
                def file_progress_callback(num_processed_in_batch):
                    nonlocal frames_done_for_file
                    if self.stop_processing.is_set():
                        raise InterruptedError("Processing stopped by user.")
                    
                    frames_done_for_file += num_processed_in_batch
                    progress = (frames_done_for_file / total_frames_in_file) * 100 if total_frames_in_file > 0 else 0
                    self.progress_queue.put({
                        "file_progress": progress,
                        "file_status": f"Frame {frames_done_for_file} of {total_frames_in_file}"
                    })
                
                try:
                    scaler = EMAMinMaxScaler(decay=decay, buffer_size=buffer_size)
                    process_video(
                        video_input=input_path,
                        video_output=output_path,
                        model=self.model,
                        process_res=resolution,
                        batch_size=batch_size,
                        progress_callback=file_progress_callback,
                        stop_event=self.stop_processing,
                        scaler=scaler,
                        use_gamma=self.use_gamma.get(),
                        gamma_value=self.gamma_value.get()
                    )
                except InterruptedError:
                    break # Stop the outer loop as well
                except Exception as e:
                    self.progress_queue.put({"error": f"Failed to process {filename}: {e}"})
                    continue # Move to the next video

            self.progress_queue.put({"processing_done": True})

        except Exception as e:
            self.progress_queue.put({"error": str(e)})

# --- End of content from gui.py ---


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthVideoGUI(root)
    root.mainloop()