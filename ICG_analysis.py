""" Latest version of individual and ensemble avergage analysis """

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.cluster.vq import vq, kmeans, whiten
from matplotlib import pyplot as plt
from typing import Callable
from dataclasses import dataclass, field
import warnings

@dataclass
class Ensemble: 
    """
    An ensemble (collection of features/signals) and related data.
    """
    features: np.ndarray = field(default_factory=lambda: np.array([])) # 2D
    ensAvg: np.ndarray = field(default_factory=lambda: np.array([])) # 1D
    sds: np.ndarray = field(default_factory=lambda: np.array([])) # 1D
    label: str = ""
    
    gen_avg = staticmethod(lambda l: np.average(l, axis=0)
                           if len(l) > 0 and l.ndim == 2 else np.array([]))
    gen_sds = staticmethod(lambda l: np.std(l, axis=0)
                           if len(l) > 0 and l.ndim == 2 else np.array([]))
    
    def __post_init__(self) -> None:
        self.calc()
    
    def calc(self) -> None:
        if len(self.features) > 0:
            self.ensAvg = self.gen_avg(self.features)
            self.sds = self.gen_sds(self.features)
        
class ICG():
    def __init__(self) -> None:
        # Plotting
        self.shape_plot = [80,25]
        self.titleSize = 50
        self.txtSize = 30
        self.colors = [
            "#2ECC71",  # green
            "#E67E22",  # orange
            "#9B59B6",  # purple
            "#1ABC9C",  # teal
            "#E74C3C",  # red
            "#F1C40F",  # yellow
            "#E91E63",  # pink
            "#FF5722",  # deep orange
        ]
        
        # Signal data
        self.fs = 1000.0 # Sampling frequency
        self.offsetL = 256
        self.offsetR = 744
    
    # Data importing
    def importCSV(self, fileName: str) -> pd.DataFrame:
        """
        Generate dataframe of individual complexes within an ensemble.
        """
        df = pd.read_csv(fileName)
        return df    
    
    # Filtering/ensembling
    def filterByBounds(
            self, features: pd.DataFrame | np.ndarray, boundsMax: float = 1.75,
            boundsMin: float = 0.33, pMin: float = 0.33, idxStart: int = 255,
            duration: int = 500
            ) -> list[Ensemble]:
        """
        Generate an Ensemble object containing left over and discarded 
        features per individual complex boundary conditions.        
        """
        if isinstance(features, pd.DataFrame):
            features = self.extractFeatures(features)
        nMin = int (pMin * len(features))
        rem = []
        
        while len(features) > nMin:
            ref = (lambda l: np.average(l, axis=0))(features)
            ref_sub = ref[idxStart:idxStart+duration]
            rBounds = np.max(ref_sub) - np.min(ref_sub)
            idxRem = -1
            error = -1.0
            
            for i, f in enumerate(features):
                f_sub = f[idxStart:idxStart+duration]
                fBounds = np.max(f_sub) - np.min(f_sub)
                
                # Anomalous bounds size found -> If error increases, store
                if fBounds > rBounds*boundsMax or fBounds < rBounds*boundsMax:
                    if fBounds > rBounds: curError = fBounds / rBounds
                    else: curError = rBounds / fBounds
                    
                    if curError > error:
                        error = curError
                        idxRem = i
                        
            # Remove most erroneous complex
            if idxRem == -1: break # No bad complex found
            rem.append(features[idxRem])
            features = np.delete(features, idxRem, axis=0)
            
        return [
            Ensemble(features=features, label="Kept signals"),
            Ensemble(features=np.array(rem), label="Removed signals")
            ]
    
    def extractFeatures(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the feature vectors out of a dataframe.
        """
        dataCols = [col for col in df.columns if col.startswith("s")]
        features = df[dataCols].to_numpy()
        return features
    
    # Plotting
    def multiPlotLinesFn(
            self, data: Ensemble, alpha: float = 0.15, lwMain: int = 6,
            lw: int = 4, color: str = "skyblue", nSignals: int = 20, 
            yrange: list[float] = None,
            title: str = "", xlab: str = "", ylab: str = ""
            ) -> Callable[[plt.Axes], None]:
        """
        Generate plotting func of signal(s) with lighter shaded ensemble.
        """
        assert nSignals <= len(data.features), "Not enough complexes to show."
        idxs = np.random.choice(len(data.features), nSignals, replace=False)
        x = np.arange(len(data.ensAvg))
        
        def plot_fn(ax: plt.Axes) -> None:
            ax.plot(x, data.ensAvg, color=color, lw=lwMain)
            
            for f in data.features[idxs]:
                ax.plot(x, f, color=color, lw=lw, alpha=alpha)
                
            ax.set_title(title, fontsize=self.txtSize)
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.txtSize)
            ax.set_xlabel(xlab, fontsize=self.txtSize)
            ax.set_ylabel(ylab, fontsize=self.txtSize)
            ax.set_xlim(x[0], x[-1])
            if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
            
        return plot_fn
            
    def multiPlotShadeFn(
            self, data: Ensemble | list[Ensemble], showSD: bool = True,  
            alpha: float = 0.15, yrange: list[float] = None, lw: int = 4,
            colors: list[str] = None, vline: int = -1, vlab: str = "", 
            title: str = "", xlab: str = "", ylab: str = "", 
            legend: bool = True
            ) -> Callable[[plt.Axes], None]:
        """
        Generate plotting func of signal(s) with optional shaded SD.
        """
        if isinstance(data, Ensemble): data = [data]
        if colors is None: colors = self.colors
        wString = f"{len(colors)} supplied for {len(data)} signals"
        assert len(colors) > len(data), wString
        
        def plot_fn(ax: plt.Axes) -> None:
            for d, color in zip(data, colors[:len(data)]):
                if len(d.ensAvg) > 0:
                    x = np.arange(len(d.ensAvg))
                    ax.plot(
                        x, d.ensAvg, color=color, lw=lw, label=d.label
                        )
                    
                if showSD:
                    ax.fill_between(
                        x, d.ensAvg-d.sds, d.ensAvg+d.sds, color=color, 
                        alpha=alpha
                        )
                
            if vline > 0:
                ax.axvline(
                    vline, color="red", lw=lw*2, linestyle=":", label=vlab
                    )
                
            ax.set_title(title, fontsize=self.txtSize)
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.txtSize)
            ax.set_xlabel(xlab, fontsize=self.txtSize)
            ax.set_ylabel(ylab, fontsize=self.txtSize)
            ax.set_xlim(x[0], x[-1])
            if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
            if legend: ax.legend(fontsize=self.txtSize)
            
        return plot_fn
    
    def plotFigs(
            self, plot_fn: Callable[[plt.Axes], None] | 
            list[Callable[[plt.Axes], None]], 
            figsize: tuple[int] = None, vert: bool = False, title: str = ""
            ) -> None:
        """
        Generate plots using a (list of) plt.Axes object(s).
        Currently only usable for 1xn or nx1 sized figures.
        """
        if not isinstance(plot_fn, list): plot_fn = [plot_fn]
        if figsize is None: 
            figsize = self.shape_plot
            if vert: figsize[1] = figsize[1]*len(plot_fn)
        
        fig = plt.figure(figsize=figsize)
        if vert: subfigs = fig.subfigures(len(plot_fn), 1)
        else: subfigs = fig.subfigures(1, len(plot_fn))
        if not isinstance(subfigs, np.ndarray): subfigs = [subfigs]
        
        for subfig, fn in zip(subfigs, plot_fn):
            ax = subfig.subplots()
            fn(ax)
            
        fig.suptitle(title, fontsize=self.titleSize)
        fig.tight_layout()
        plt.show()
    
    # Runs
    ...
            
icg = ICG()
df = icg.importCSV("Data/baseline_climbing.csv")





















