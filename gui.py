import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import threading
import queue
import torch
from depth_anything_3.api import DepthAnything3
from create_depth_video import process_video

class DepthVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Video Creator")
        self.config_file = "gui_config.json"
        
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self.progress_queue = queue.Queue()

        self.model = None
        self.device = None

        self.setup_ui()
        self.load_settings()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_queue()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Input/Output Folders ---
        path_frame = ttk.LabelFrame(main_frame, text="Folders", padding="10")
        path_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
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
        settings_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Processing Resolution:").grid(row=0, column=0, sticky=tk.W)
        self.resolution = tk.IntVar(value=504)
        ttk.Spinbox(settings_frame, from_=252, to=1008, increment=252, textvariable=self.resolution, width=10).grid(row=0, column=1, sticky=tk.W)

        # --- Progress and Status ---
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))
        progress_frame.columnconfigure(0, weight=1)

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
        button_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing_thread, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        for frame in [path_frame, settings_frame, progress_frame]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=2)

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
                    self.resolution.set(config.get("resolution", 504))
        except (json.JSONDecodeError, IOError) as e:
            messagebox.showerror("Error", f"Could not load settings: {e}")

    def save_settings(self):
        config = {
            "input_folder": self.input_folder.get(),
            "output_folder": self.output_folder.get(),
            "resolution": self.resolution.get()
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
                def file_progress_callback(current_frame, total_frames):
                    if self.stop_processing.is_set():
                        raise InterruptedError("Processing stopped by user.")
                    progress = (current_frame / total_frames) * 100
                    self.progress_queue.put({
                        "file_progress": progress,
                        "file_status": f"Frame {current_frame} of {total_frames}"
                    })
                
                try:
                    process_video(
                        video_input=input_path,
                        video_output=output_path,
                        model=self.model,
                        process_res=resolution,
                        progress_callback=file_progress_callback,
                        stop_event=self.stop_processing
                    )
                except InterruptedError:
                    break # Stop the outer loop as well
                except Exception as e:
                    self.progress_queue.put({"error": f"Failed to process {filename}: {e}"})
                    continue # Move to the next video

            self.progress_queue.put({"processing_done": True})

        except Exception as e:
            self.progress_queue.put({"error": str(e)})

if __name__ == "__main__":
    root = tk.Tk()
    app = DepthVideoGUI(root)
    root.mainloop()
