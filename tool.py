import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf


AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

def collect_audio_files(folder_path):
    """
    Recursively collect all audio files under the given folder,
    matching AUDIO_EXTENSIONS.
    Returns a sorted list of full file paths.
    """
    audio_paths = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(AUDIO_EXTENSIONS):
                fpath = os.path.join(root, fname)
                audio_paths.append(fpath)
    return sorted(audio_paths)

class AudioAnalyzerOnDemand(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Analyzer (On-Demand Calculation)")

        # ─────────────────────────────────────────
        # 1) FOLDER + ANALYSIS FRAME
        # ─────────────────────────────────────────
        folder_frame = tk.Frame(self)
        folder_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.select_folder_btn = tk.Button(
            folder_frame, 
            text="Select Folder", 
            command=self.select_folder
        )
        self.select_folder_btn.pack(side=tk.LEFT, padx=5)

        self.analysis_var = tk.StringVar(value="Spectrogram")
        self.analysis_menu = tk.OptionMenu(folder_frame, self.analysis_var, "Spectrogram", "FFT")
        self.analysis_menu.pack(side=tk.LEFT, padx=5)

        self.scan_btn = tk.Button(folder_frame, text="Scan Folder", command=self.scan_folder)
        self.scan_btn.pack(side=tk.LEFT, padx=5)

        # No separate "Analyze All" button now; we do on-demand.

        self.selected_folder = None

        # ─────────────────────────────────────────
        # 2) STFT/FFT PARAMETERS FRAME
        # ─────────────────────────────────────────
        param_frame = tk.Frame(self)
        param_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(param_frame, text="Window:").pack(side=tk.LEFT, padx=5)
        self.window_var = tk.StringVar(value="hann")
        window_options = ["hann", "hamming", "blackman", "bartlett", "kaiser"]
        tk.OptionMenu(param_frame, self.window_var, *window_options).pack(side=tk.LEFT)

        tk.Label(param_frame, text="n_fft:").pack(side=tk.LEFT, padx=5)
        self.nfft_var = tk.StringVar(value="2048")
        tk.Entry(param_frame, textvariable=self.nfft_var, width=6).pack(side=tk.LEFT)

        tk.Label(param_frame, text="hop_length:").pack(side=tk.LEFT, padx=5)
        self.hop_var = tk.StringVar(value="512")
        tk.Entry(param_frame, textvariable=self.hop_var, width=6).pack(side=tk.LEFT)

        # ─────────────────────────────────────────
        # 3) FFT LOG SCALE OPTIONS
        # ─────────────────────────────────────────
        log_frame = tk.Frame(self)
        log_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.fft_logfreq_var = tk.BooleanVar(value=False)
        self.fft_logamp_var = tk.BooleanVar(value=False)

        tk.Checkbutton(
            log_frame, 
            text="FFT Log-Frequency Axis", 
            variable=self.fft_logfreq_var
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Checkbutton(
            log_frame, 
            text="FFT Log-Amplitude Axis", 
            variable=self.fft_logamp_var
        ).pack(side=tk.LEFT, padx=10)

        # ─────────────────────────────────────────
        # 4) AXIS LIMITS FRAME
        # ─────────────────────────────────────────
        axis_frame = tk.Frame(self)
        axis_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(axis_frame, text="X min:").pack(side=tk.LEFT, padx=2)
        self.xmin_var = tk.StringVar(value="")
        tk.Entry(axis_frame, textvariable=self.xmin_var, width=6).pack(side=tk.LEFT)

        tk.Label(axis_frame, text="X max:").pack(side=tk.LEFT, padx=2)
        self.xmax_var = tk.StringVar(value="")
        tk.Entry(axis_frame, textvariable=self.xmax_var, width=6).pack(side=tk.LEFT)

        tk.Label(axis_frame, text="   Y min:").pack(side=tk.LEFT, padx=2)
        self.ymin_var = tk.StringVar(value="")
        tk.Entry(axis_frame, textvariable=self.ymin_var, width=6).pack(side=tk.LEFT)

        tk.Label(axis_frame, text="Y max:").pack(side=tk.LEFT, padx=2)
        self.ymax_var = tk.StringVar(value="")
        tk.Entry(axis_frame, textvariable=self.ymax_var, width=6).pack(side=tk.LEFT)

        tk.Label(axis_frame, text="(Leave blank for auto)").pack(side=tk.LEFT, padx=5)

        # ─────────────────────────────────────────
        # 5) FILE LIST + PLOT BUTTON
        # ─────────────────────────────────────────
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        list_frame = tk.Frame(bottom_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(list_frame, text="Audio Files Found:").pack(anchor="w")
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, width=50, height=15)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(bottom_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="Actions:").pack(anchor="w")
        self.plot_btn = tk.Button(right_frame, text="Plot Selected", command=self.plot_selected)
        self.plot_btn.pack(pady=5, anchor="w")

        # ─────────────────────────────────────────
        # DATA STORAGE
        # ─────────────────────────────────────────
        self.file_paths = []
        # We store audio in a dict if it's loaded
        self.audio_data_dict = {}     # fpath -> (audio, sr)
        self.spectrogram_dict = {}    # fpath -> (S_db, sr, hop, window)
        self.fft_dict = {}            # fpath -> (freq_pos, mag_pos, sr)

    # ─────────────────────────────────────────
    #  FOLDER SELECTION & SCANNING
    # ─────────────────────────────────────────
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Audio Files")
        if folder:
            self.selected_folder = folder
            print(f"Folder selected: {folder}")

    def scan_folder(self):
        """Recursively find audio files and list them."""
        if not self.selected_folder:
            print("No folder selected.")
            return

        self.file_listbox.delete(0, tk.END)
        self.file_paths.clear()
        self.audio_data_dict.clear()
        self.spectrogram_dict.clear()
        self.fft_dict.clear()

        found_files = collect_audio_files(self.selected_folder)
        self.file_paths = found_files

        for fpath in found_files:
            base_name = os.path.relpath(fpath, self.selected_folder)
            self.file_listbox.insert(tk.END, base_name)

        print(f"Found {len(found_files)} audio files under {self.selected_folder}.")

    # ─────────────────────────────────────────
    #  PLOT SELECTED FILES (AUTOMATIC COMPUTE)
    # ─────────────────────────────────────────
    def plot_selected(self):
        """When user clicks 'Plot Selected', automatically compute & display."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            print("No files selected!")
            return

        selected_fpaths = [self.file_paths[i] for i in selected_indices]
        analysis_mode = self.analysis_var.get()

        # We'll parse STFT parameters now
        try:
            n_fft = int(self.nfft_var.get())
            hop_length = int(self.hop_var.get())
        except ValueError:
            print("Invalid n_fft or hop_length. Must be integers.")
            return

        if analysis_mode == "Spectrogram":
            self.plot_spectrograms(selected_fpaths, n_fft, hop_length)
        elif analysis_mode == "FFT":
            self.plot_ffts(selected_fpaths)
        else:
            print(f"Unknown analysis mode: {analysis_mode}")

    # ─────────────────────────────────────────
    #  SPECTROGRAM PLOTTING
    # ─────────────────────────────────────────
    def plot_spectrograms(self, fpaths, n_fft, hop_length):
        """Compute spectrogram if needed, then plot each file in its own figure."""
        x_min, x_max, y_min, y_max = self.parse_axis_limits()

        for fpath in fpaths:
            # If we haven't computed a spectrogram, do so now
            if fpath not in self.spectrogram_dict:
                self.compute_spectrogram(fpath, n_fft, hop_length)

            if fpath not in self.spectrogram_dict:
                # If still not present, some error must have occurred
                continue

            S_db, sr, hop_l, window_type = self.spectrogram_dict[fpath]
            base_name = os.path.relpath(fpath, self.selected_folder)

            plt.figure(figsize=(10, 6))
            librosa.display.specshow(
                S_db,
                sr=sr,
                hop_length=hop_l,
                x_axis='time',
                y_axis='log'
            )
            plt.title(f"Spectrogram: {base_name}\n(window={window_type}, n_fft={n_fft}, hop={hop_l})")
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")

            # Axis limits (time/frequency)
            if x_min is not None or x_max is not None:
                plt.xlim(x_min, x_max)
            if y_min is not None or y_max is not None:
                plt.ylim(y_min, y_max)

        plt.show()

    def compute_spectrogram(self, fpath, n_fft, hop_length):
        """Load audio if necessary, compute STFT -> dB, store in dict."""
        if fpath not in self.audio_data_dict:
            self.load_audio(fpath)
        if fpath not in self.audio_data_dict:
            # If load failed
            return

        audio, sr = self.audio_data_dict[fpath]
        window_type = self.window_var.get()

        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window_type)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        self.spectrogram_dict[fpath] = (S_db, sr, hop_length, window_type)

    # ─────────────────────────────────────────
    #  FFT PLOTTING
    # ─────────────────────────────────────────
    def plot_ffts(self, fpaths):
        """Compute FFT if needed, then plot all selected on one figure."""
        x_min, x_max, y_min, y_max = self.parse_axis_limits()
        log_freq = self.fft_logfreq_var.get()
        log_amp = self.fft_logamp_var.get()

        plt.figure(figsize=(10, 6))

        for fpath in fpaths:
            if fpath not in self.fft_dict:
                self.compute_fft(fpath)

            if fpath not in self.fft_dict:
                continue

            freq_pos, mag_pos, sr = self.fft_dict[fpath]
            base_name = os.path.relpath(fpath, self.selected_folder)

            # If using log frequency, skip the DC bin
            if log_freq:
                freq_plot = freq_pos[1:]
                mag_plot = mag_pos[1:]
            else:
                freq_plot = freq_pos
                mag_plot = mag_pos

            # If using log amplitude, clip near zero
            if log_amp:
                mag_plot = np.maximum(mag_plot, 1e-12)

            plt.plot(freq_plot, mag_plot, label=base_name)

        plt.title("FFT of Selected Files")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # Set log scales if requested
        if log_freq:
            plt.xscale('log')
        if log_amp:
            plt.yscale('log')

        # Axis limits (frequency/amplitude)
        if x_min is not None or x_max is not None:
            plt.xlim(x_min, x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(y_min, y_max)

        plt.show()

    def compute_fft(self, fpath):
        """Load audio if needed, then do single-sided FFT."""
        if fpath not in self.audio_data_dict:
            self.load_audio(fpath)
        if fpath not in self.audio_data_dict:
            return

        audio, sr = self.audio_data_dict[fpath]
        N = len(audio)
        if N < 2:
            print(f"Audio too short for FFT: {fpath}")
            return

        fft_complex = np.fft.fft(audio)
        freq = np.fft.fftfreq(N, d=1.0/sr)

        half_N = N // 2
        freq_pos = freq[:half_N]
        magnitude_pos = np.abs(fft_complex[:half_N]) * 2.0 / N

        self.fft_dict[fpath] = (freq_pos, magnitude_pos, sr)

    # ─────────────────────────────────────────
    #  LOAD AUDIO
    # ─────────────────────────────────────────
    # def load_audio(self, fpath):
    #     """Try loading audio with librosa, store in audio_data_dict."""
    #     try:
    #         audio, sr = librosa.load(fpath, sr=None, mono=True)
    #         self.audio_data_dict[fpath] = (audio, sr)
    #     except Exception as e:
    #         print(f"Error loading {fpath}: {e}")
            
    def load_audio(self, fpath):
        """Load audio with SoundFile to preserve original amplitude."""
        try:
            audio, sr = sf.read(fpath, dtype = 'float32')
            
            # audio_max = np.max(np.abs(audio))
            # if audio_max > 0:
            #     audio = audio / audio_max
                
            self.audio_data_dict[fpath] = (audio, sr)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    # ─────────────────────────────────────────
    #  PARSE AXIS LIMITS
    # ─────────────────────────────────────────
    def parse_axis_limits(self):
        """
        Attempt to parse x_min, x_max, y_min, y_max from user entries.
        Return None for any invalid or empty entry.
        """
        def parse_float(text):
            text = text.strip()
            if text == "":
                return None
            try:
                return float(text)
            except ValueError:
                return None

        x_min = parse_float(self.xmin_var.get())
        x_max = parse_float(self.xmax_var.get())
        y_min = parse_float(self.ymin_var.get())
        y_max = parse_float(self.ymax_var.get())

        return x_min, x_max, y_min, y_max

if __name__ == "__main__":
    app = AudioAnalyzerOnDemand()
    app.mainloop()