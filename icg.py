import pandas as pd
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.cluster.vq import vq, kmeans, whiten
from dataclasses import dataclass
import wfdb
import json

path = "data/"
shape_single = (40,15)
       
def lowpass_filter(sig: npt.NDArray[np.float_], cutoff_hz, fs_hz, order=4):
    """
    Butterworth filter
    """
    nyq = fs_hz / 2
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, sig) 

def MAS(sig: npt.NDArray[np.float_], window: int = 5) -> None:
    """
    Mean Average Smoothing
    """
    out = np.zeros([len(sig)])
    for i in range(window, len(sig)-window):
        out[i] = np.average(sig[i-window:i+window])
    return out

def cDiff(sig: npt.NDArray[np.float_]) -> None:
    """
    Differentiation using center difference method.
    """
    out = np.zeros([len(sig)])
    for i in range(1, len(sig)-1):
        out[i] = (sig[i+1] - sig[i-1]) / 2
    return out
    
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
   
    # PLOTTING
    def iplot(self, sig: npt.NDArray[np.float_] | list[npt.NDArray[np.float_]],
         figTitle: str = "", title: str | list[str] = "", bounds: bool = True, 
         sd: bool = False, lw: int = 4, fontsize: int = 40) -> None:
        """
        Quick plotting func.
        """
        if isinstance(sig, np.ndarray): sig = [sig]
        if isinstance(title, str): title = [title for _ in range(len(sig))]
        fig, axs = plt.subplots(nrows=len(sig),
                               figsize=(self.shape_plot[0], 
                                        self.shape_plot[1] * len(sig))
                               )
        fig.suptitle(figTitle, fontsize=fontsize)
        
        x = np.arange(0, len(sig[0]), 1)      
        t_maxs = find_peaks(sig[0])[0] # Simple C-point estimator
        t_maxs = t_maxs[t_maxs > self.offsetL]
        if (sig[0][t_maxs[0]] >= 1.4 * sig[0][t_maxs[0]]): t_max = t_maxs[1]
        else: t_max = t_maxs[0] 
        
        for i, s in enumerate(sig):
            if isinstance(axs, plt.Axes): ax = axs
            else: ax = axs[i]
            ax.plot(x, s, lw=lw)
            ax.tick_params(axis="both", labelsize=30)
            ax.grid(which="both", axis="both")
            ax.set_xlim(x[0], x[-1])
            ax.set_title(title[i], fontsize=fontsize)
            ax.yaxis.get_offset_text().set_fontsize(fontsize)
            if bounds:
                ax.axvspan(self.offsetL, t_max, color="orange", alpha=0.3)
    
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
             bLeft: int = -1, bDelta: int = -1, 
             figTitle: str = "", title: str | list[str] = "", sd: bool = False, 
             lw: int = 4, fontsize: int = 40) -> None:
        """
        Quick plotting func.
        """
        if isinstance(sig, np.ndarray): sig = [sig]
        if isinstance(title, str): title = [title for _ in range(len(sig))]
        fig, axs = plt.subplots(nrows=len(sig),
                               figsize=(self.shape_plot[0], 
                                        self.shape_plot[1] * len(sig))
                               )
        fig.suptitle(figTitle, fontsize=fontsize)
        
        x = np.arange(0, len(sig[0]), 1)                    
        for i, s in enumerate(sig):
            if isinstance(axs, plt.Axes): ax = axs
            else: ax = axs[i]
            if bLeft >= 0:
                ax.axvspan(bLeft, bLeft+bDelta, color="orange", alpha=0.3)
            ax.plot(x, s, lw=lw)
            ax.tick_params(axis="both", labelsize=30)
            ax.grid(which="both", axis="both")
            ax.set_xlim(x[0], x[-1])
            ax.set_title(title[i], fontsize=fontsize)
            ax.yaxis.get_offset_text().set_fontsize(fontsize)
        plt.tight_layout()
    
    def plotSD(self, ens: npt.NDArray[np.float_], x_start: int = -1, 
               title: str = "", lw: int = 4, fontsize: int = 40) -> None:
        sig = np.average(ens, axis=0)
        sd = np.std(ens, axis=0)
        fig, ax = plt.subplots(figsize=(
            self.shape_plot[0], self.shape_plot[1])
            )
        if x_start < 0: x_start = 0
        x = np.arange(x_start, x_start + len(sig), 1)
        ax.fill_between(x, sig-sd, sig+sd, alpha=0.25)
        ax.plot(x, sig, lw=lw)
        ax.tick_params(axis="both", labelsize=30)
        ax.grid(which="both", axis="both")
        ax.set_xlim(x[0], x[-1])
        ax.set_title(title, fontsize=fontsize)
        
    def plotMult(self, sig: npt.NDArray[np.float_] | 
                     list[npt.NDArray[np.float_]],
                     sd: npt.NDArray[np.float_] = None,
                     col: str | list[str] = "skyblue", figTitle: str = "",
                     fontsize: int = 40, lw: int = 4) -> None:
        if isinstance(sig, np.ndarray): 
            sig = [sig]
            col = [col]
        else: 
            if (len(col) < len(sig)): col = [col[0] for _ in range(len(sig))]
        
        x = np.arange(0, len(sig[0]), 1)
        fig, ax = plt.subplots(
                               figsize=(self.shape_plot[0], self.shape_plot[1])
                               )
        fig.suptitle(figTitle, fontsize=fontsize)

        for i, s in enumerate(sig):
            ax.plot(x, s, color=col[i], lw=lw, alpha=0.3)
            ax.tick_params(axis="both", labelsize=30)
            ax.grid(which="both", axis="both")
            ax.set_xlim(x[0], x[-1])
        if isinstance(sd, np.ndarray):
            ax.plot(x, np.average(sd, axis=0), col[0], lw=2*lw)
        plt.tight_layout()            
    
    def heatmap(self, sigs: npt.NDArray[np.float_], title: str = "", 
                fontsize: int = 40, dpi: int = 250, cmap: str = "magma", 
                showPeak: bool = False, peakPos: int = -1) -> None:
        """
        Heatmap plotting func.
        """
        h, w = sigs.shape[:2]
        fig, ax = plt.subplots(figsize=(30,12), dpi=dpi)
        im = ax.imshow(sigs, aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel("Time wrt R-Peak [ms]", fontsize=fontsize)
        ax.set_ylabel("Signal index", fontsize=fontsize)
        if showPeak:
            if peakPos == -1: peakPos = self.offsetL
            ax.axvline(x=peakPos, color="red", linewidth=3, 
                       label="R-peak position")
            ax.legend(fontsize=fontsize)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("dZ/dT [Ω/s]", fontsize=fontsize)
        plt.tight_layout()

    # RUNS
    def run_std(self, peaksFile: str, ecg: npt.NDArray[np.float_], 
                dzdt: npt.NDArray[np.float_], flip: bool = True, 
                cutoff: int = 100, show: bool = True, hmap: bool = False, 
                inter: bool = True, out: bool = False) -> None:
        """
        Run standard ensemble averaging (ensemble dzdt) and plotting funcs.

        Parameters
        ----------
        peaksFile : JSON containing peak times for a label.
        cutoff : Cutoff frequency for the low-pass filter of the third derivative of Z0.
        hmap : Output signal heatmap instead of ensembled plot.
        inter : Use interactive plotting tool (QT5).
        out : Output the input signal and the ensemble average signal.
        plotTitle : Name of the plot(s)
        """
        if flip: ecg *= -1
        times = icg.importJson(path+peaksFile+".json", convert=True)
        idxs = self.findIdxs(ticks, times)
        ensembles_dzdt = self.ensAvg(dzdt, idxs)
        ensembles_ecg = self.ensAvg(ecg, idxs)
        
        ecg = np.average(ensembles_ecg, axis=0)
        avg = np.average(ensembles_dzdt, axis=0)
        davg = np.gradient(avg)
        ddavg = np.gradient(davg)
        ddavg_filt = lowpass_filter(ddavg, cutoff, self.fs)
        dddavg = np.gradient(ddavg_filt)
        sigs = [avg, ddavg_filt, dddavg]
        titles = [r"$\frac{dZ}{dT}$",
                  r"$\frac{d^{3}Z}{dT^3}$ (Low-pass till 100 Hz)", 
                  r"$\frac{d^{4}Z}{dT^4}$"]
        
        if show:
            if not hmap: 
                if inter: self.iplot(sigs, figTitle=peaksFile, title=titles)
                else: self.plot(sigs, figTitle=peaksFile, title=titles)
            else: self.heatmap(sigs, title=peaksFile)
        
        if out: return [ensembles_dzdt, avg]
        
    def run_z(self, peaksFile: str, cutoff: int = 100, show: bool = True,
              hmap: bool = False, inter: bool = True, out: bool = False
              ) -> None:
        """
        Run ensemble averaging on Z0 first before differentiation.
        """
        times = icg.importJson(path+peaksFile+".json", convert=True)
        idxs = self.findIdxs(ticks, times)
        ensembles = self.ensAvg(dz, idxs)
        
        if show:
            if not hmap: 
                if inter: self.iplot(sigs, figTitle=peaksFile, title=titles)
                else: self.plot(sigs, figTitle=peaksFile, title=titles)
            else: self.heatmap(sigs, title=peaksFile)
        
        if out: return [ensembles, avg]
        
    def run_compare(self, peaksFile: str, cutoff: int = 100, hmap: bool = False, 
            inter: bool = True, out: bool = False) -> None:
        """
        Compare ensemble averaging of Z0 and DZDT.
        """
        sigs = [
            self.run_std(peaksFile, show=False, out=True)[1],
            self.run_z(peaksFile, show=False, out=True)[1]
            ]
        titles = ["Ensemble of DZDT", "Ensemble of Z0"]
        
        self.iplot(sigs, peaksFile, titles)
        
        if out: return sigs
    
    # ENSEMBLE AVERAGING    
    def ens(self, data: npt.NDArray[np.float_], idxs: list[int], 
            offsetL: int = 50, offsetR: int = 200) -> npt.NDArray[np.float_]:
        """
        Collect ensembles with custom left/right offsets.
        """
        ensembles = np.array(
            [data[i-offsetL:i+offsetR] for i in idxs]
            )
        
        return ensembles
    
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
    def importJson(self, name: str, convert: bool = True
                   ) -> npt.NDArray[np.float_]:
        with open(name, 'r') as f:
            data = json.load(f)
            if convert: data = [int(d) for d in data]
            return data
        
    def importTxt(self, name: str, convert: bool = True
                  ) -> npt.NDArray[np.float_]:
        with open(name, 'r') as f:
            data = f.read().split()
            if convert: data = np.array([float(d) for d in data])
            return data
    
    def _importBin(self, name: str, dtype: str = "int32",
              lower_bound: int = -2147483648, upper_bound: int = 2147483648,
              lower_value: float = -8192, upper_value: float = 8192,
              fs: int = 1000, flip: bool = False
              ) -> npt.NDArray[np.float64]:
        """
        Import binary signal data using the same linear calibration as BinaryFile.java.
        
        lower_bound / upper_bound : digital min/max (from nBits, e.g. -2^15 .. 2^15-1)
        lower_value / upper_value : physical min/max (lMinValue/lMaxValue / lMinMaxDivider)
        """
        self.fs = fs
    
        data_types = {
            'float32': np.float32,
            'float64': np.float64,
            'int16':   np.int16,
            'int32':   np.int32,
            'uint16':  np.uint16,
        }
    
        data = np.fromfile(name, dtype=data_types[dtype])
        # Java reads with a standard (big-endian on some systems) FileChannel —
        # only byteswap if your platform endianness differs from the recorder
        # data.byteswap(inplace=True)  # enable if needed
    
        data = data.astype(np.float64)
    
        # Replicate Java's getRealValueFromSampleValue():
        #   realSlope    = (upperValue - lowerValue) / (upperBound - lowerBound)
        #   realConstant = lowerValue - realSlope * lowerBound
        #   physicalVal  = realSlope * sampleValue + realConstant
        if all(v is not None for v in [lower_bound, upper_bound, lower_value, upper_value]):
            slope    = (upper_value - lower_value) / (upper_bound - lower_bound)
            constant = lower_value - slope * lower_bound
            data     = data / slope + constant
        else:
            raise ValueError("Provide all four calibration parameters.")
    
        if flip:
            data *= -1
    
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
path = "Data/CLIMBING_ELI/"

times = icg.importJson("data/CLIMBING_ELI/peaks_1.json")
ticks = icg.importBin("Data/CLIMBING_ELI/TicksA.bin")
ecg = icg.importBin("Data/CLIMBING_ELI/FILTECG.bin")
dzdt = icg.importBin("Data/CLIMBING_ELI/FILTDZDT.bin") # First derivative of ICG
dz = icg.importBin("Data/CLIMBING_ELI/DZ.bin") # ICG data

idxs = icg.findIdxs(ticks, times)
ensembles = icg.ensAvg(dzdt, idxs)
avg = np.average(ensembles, axis=0)
davg = np.gradient(avg)
ddavg = np.gradient(davg)
sigs = [avg, davg, ddavg, lowpass_filter(ddavg, 100, 1000)]
titles = ["avg", "d_avg", "dd_avg", "dd_avg lowpass till 100 Hz"]

# Magic commands
# %matplotlib qt5
# %matplotlib inline

