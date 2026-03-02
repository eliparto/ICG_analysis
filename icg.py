import pandas as pd
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from dataclasses import dataclass
import wfdb
import json

path = "data/"
shape_single = (40,15)
       
def lowpass_filter(sig, cutoff_hz, fs_hz, order=4):
    nyq = fs_hz / 2
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, sig) 
    
class ICG():
    """
    Analysis of ICG signals.
    """
    def __init__(self, path: str = "") -> None:
        if path != "": self.path = path # Allow usage of functions within other objects
        
        # Plotting
        self.shape_plot = [80,15]
        # Signal data
        self.fs = 1000.0 # Sampling frequency
        self.offsetL = 256
        self.offsetR = 744

    # SIGNAL PROCESSING    
    def findTickIndex(self) -> int:
        """
        Find the index in a dataset corresponding to the tick of a time-point
        """
        ...
   
    # PLOTTING
    def iplot(self, sig: npt.NDArray[np.float_] | list[npt.NDArray[np.float_]],
         tit: str | list[str] = "", bounds: bool = True, sd: bool = False,
         lw: int = 8, fontsize: int = 20) -> None:
        """
        Quick plotting func.
        """
        if isinstance(sig, np.ndarray): sig = [sig]
        if isinstance(tit, str): tit = [tit for _ in range(len(sig))]
        fig, axs = plt.subplots(nrows=len(sig),
                               figsize=(self.shape_plot[0], 
                                        self.shape_plot[1] * len(sig))
                               )
        
        x = np.arange(0, len(sig[0]), 1)      
        t_maxs = find_peaks(sig[0])[0] # Simple C-point estimator
        t_maxs = t_maxs[t_maxs>self.offsetL]
        if (sig[0][t_maxs[1]] >= 1.4 * sig[0][t_maxs[0]]): t_max = t_maxs[1]
        else: t_max = t_maxs[0] 
        
        for i, s in enumerate(sig):
            if isinstance(axs, plt.Axes): ax = axs
            else: ax = axs[i]
            ax.plot(x, s, lw=lw)
            ax.tick_params(axis="both", labelsize=30)
            ax.grid(which="both", axis="both")
            ax.set_xlim(x[0], x[-1])
            ax.set_title(tit[i], fontsize=fontsize)
            if bounds:
                ax.axvspan(self.offsetL, t_max, color="orange", alpha=0.3)
    
        # --- Draggable vertical line across all axes ---
        all_axes = [axs] if isinstance(axs, plt.Axes) else list(axs)
        x_pos = [x[len(x) // 2]]  # mutable container so callbacks can update it
        
        vlines = [ax.axvline(x_pos[0], color='r', linestyle='--', lw=2, label='B-point') for ax in all_axes]
        all_axes[0].legend(fontsize=20)
    
        dragging = [False]
    
        def on_press(event):
            if event.inaxes not in all_axes or event.xdata is None:
                return
            xlim = event.inaxes.get_xlim()
            threshold = (xlim[1] - xlim[0]) * 0.01
            if abs(event.xdata - x_pos[0]) < threshold:
                dragging[0] = True
    
        def on_motion(event):
            if not dragging[0] or event.inaxes not in all_axes or event.xdata is None:
                return
            x_pos[0] = event.xdata
            for vl in vlines:
                vl.set_xdata([x_pos[0], x_pos[0]])
            fig.canvas.draw_idle()
    
        def on_release(event):
            if dragging[0]:
                dragging[0] = False
                # Snap to nearest sample
                idx = int(np.clip(np.round(x_pos[0]), x[0], x[-1]))
                x_pos[0] = idx
                for vl in vlines:
                    vl.set_xdata([x_pos[0], x_pos[0]])
                fig.canvas.draw_idle()
                print(f"B-point placed at sample index: {idx}")
                self.b_point = idx  # store on self if you want it later
    
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
    
        plt.tight_layout()
        plt.show()
    
    def plot(self, sig: npt.NDArray[np.float_] | list[npt.NDArray[np.float_]],
             tit: str | list[str] = "", sd: bool = False, lw: int = 8, 
             fontsize: int = 20) -> None:
        """
        Quick plotting func.
        """
        if isinstance(sig, np.ndarray): sig = [sig]
        if isinstance(tit, str): tit = [tit for _ in range(len(sig))]
        fig, axs = plt.subplots(nrows=len(sig),
                               figsize=(self.shape_plot[0], 
                                        self.shape_plot[1] * len(sig))
                               )
        
        x = np.arange(0, len(sig[0]), 1)                    
        for i, s in enumerate(sig):
            if isinstance(axs, plt.Axes): ax = axs
            else: ax = axs[i]
            ax.plot(x, s, lw=lw)
            ax.tick_params(axis="both", labelsize=30)
            ax.grid(which="both", axis="both")
            ax.set_xlim(x[0], x[-1])
            ax.set_title(tit[i], fontsize=fontsize)
    
    def heatmap(self, sigs: npt.NDArray[np.float_], figName: str = "", 
                fontsize: int = 20, dpi: int = 250, cmap: str  ="viridis"
                ) -> None:
        """
        Heatmap plotting func.
        """
        h, w = sigs.shape[:2]
        fig, ax = plt.subplots(figsize=(30,12), dpi=dpi)
        im = ax.imshow(sigs, aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_title(figName, fontsize=fontsize)
        ax.set_xlabel("Time wrt R-Peak [ms]", fontsize=fontsize)
        ax.set_ylabel("Signal index", fontsize=fontsize)    
        ax.axvline(x=self.offsetL, color="red", linewidth=3, 
                   label="R-peak position")
        ax.legend(fontsize=fontsize)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("dZ/dT [Ω/s]", fontsize=fontsize)
        plt.tight_layout()

    # ENSEMBLE AVERAGING
    def ensAvg(self, data: npt.NDArray[np.float_], idxs: list[int]) -> None:
        """
        Generate the ensemble averaged DZDT signal.
        """
        ensembles = np.array(
            [data[i-self.offsetL:i+self.offsetR] for i in idxs]
            )
        
        return ensembles
            
    def findIdxs(self, ticks: list[float], times: list[int]
                 ) -> list[int]:
        """
        Find the indices corresponding to peak times in the recorded data
        """
        idxs = []
        curIdx = 0
        ticks = [int(t*1000) for t in ticks] # Reformat into compatible form
        
        for num, time in enumerate(times):
            for i in range(curIdx, len(ticks)):
                if ticks[i] == time: # Index of time-point found  
                    idxs.append(i)
                    curIdx = i
                    break
                
        return idxs


    # IMPORTING
    def importPeaks(self, name: str) -> None:
        """
        Import peak data to e.g. cross-ref ECG and ICG data
        """
        ...
    
    def importJson(self, name: str, convert: bool = True
                   ) -> npt.NDArray[np.float_]:
        with open(name, 'r') as f:
            data = json.load(f)
            if convert: data = [int(d) for d in data]
            return data
    
    def importBin(self, name: str, dtype: str = "int32", valRange: int = 1000,
                  fs: int = 1000, flip: bool = False
                  ) -> npt.NDArray[np.float_]:
        """
        Import ICG data stored in .bin format
        """
        self.fs = fs
        
        # Try different data types
        data_types = {
            'float32': np.float32,
            'float64': np.float64,
            'int16': np.int16,
            'int32': np.int32,
            'uint16': np.uint16,
        }
        
        # Read the binary file
        data = np.fromfile(name, dtype=data_types[dtype])
        data = data.byteswap()
        if abs(min(data)) > max(data): data *= -1
        if flip: data *= -1
        data = data.astype(float) / valRange
        
        return data
###############################################################################
# SETUP
icg = ICG(path=path)

times = icg.importJson("data/CLIMBING_ELI/peakTimes.json")
ticks = icg.importBin("Data/CLIMBING_ELI/TicksA.bin")
ecg = icg.importBin("Data/CLIMBING_ELI/FILTECG.bin")
dzdt = icg.importBin("Data/CLIMBING_ELI/FILTDZDT.bin")
titles = ["avg", "d_avg", "dd_avg", "dd_avg lowpass till 100 Hz"]

idxs = icg.findIdxs(ticks, times)
ensembles = icg.ensAvg(dzdt, idxs)
avg = np.average(ensembles, axis=0)
davg = np.gradient(avg)
ddavg = np.gradient(davg)
sigs = [avg, davg, ddavg, lowpass_filter(ddavg, 100, 1000)]

# Magic commands
# %matplotlib qt5
# %matplotlib inline