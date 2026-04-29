import numpy as np
from matplotlib import pyplot as plt
from typing import Callable

from DataFormats import Ensemble

class ICGPlot():
    def __init__(self) -> None:
        self.figsize = (80,20)
        self.fontsize = 50
        self.titlesize = 30
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
        
    # plotting functions
    def plotUnderlayFn(
            self, data: Ensemble, nSignals: int, alpha: float = 0.15, 
            lwMain: int = 6, lw: int = 4, color: str = "darkblue", 
            yrange: list[float] = None, vline: int = -1, vlab: str = "", 
            title: str = "", xlab: str = "", ylab: str = ""
            ) -> Callable[[plt.Axes], None]:
        """
        Plot a function with underlying lighter shaded functions.
        """
        if colors is None: color = self.colors[0]
        if nSignals > len(data.features): nSignals = len(data.features)
        sigIdxs = np.random.choice(len(data.features), nSignals, replace=False)
        
        def plot_fn(ax: plt.Axes) -> None:
            x = np.arange(len(data.ensAvg))
            ax.plot(x, data.ensAvg, color=color, lw=lwMain)
            
            for sig in data.features[sigIdxs]:
                ax.plot(x, sig, color=color, lw=lw, alpha=alpha)
                
            if vline > 0:
                ax.axvline(
                    vline, color="red", lw=lw*2, linestyle=":", label=vlab
                    )
                
            ax.set_title(title, fontsize=self.fontsize)
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.set_xlim(x[0], x[-1])
            if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
            
        return plot_fn
            
    def multiPlotFn(
            self, data: list[Ensemble], colors: list[str] = None, lw: int = 4,
            dy: float = 0.0, vline: int = -1, vlab: str = "", 
            title: str = "", xlab: str = "", ylab: str = "",
            legend: bool = True
            ) -> Callable[[plt.Axes], None]:
        """
        Plot multiple functions with visual settings for zoom, separation etc.
        """
        if colors is None or len(colors) < len(data): colors=self.colors
        
        def plot_fn(ax: plt.Axes) -> None:
            sum_dy = 0.0
            for d, color in zip(colors):
                x = np.arange(len(d.ensAvg))
                ax.plot(x, d.ensAvg+dy, color=color, lw=lw, label=d.label)
                sum_dy += dy
            
            if vline > 0:
                ax.axvline(
                    vline, color="red", lw=lw*2, linestyle=":", label=vlab
                    )    
            
            ax.set_title(title, fontsize=self.fontsize)
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.set_xlim(x[0], x[-1])
            if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
            if legend: ax.legend(fontsize=self.fontsize)
            
        return plot_fn
        
    def plotFn(
            self, data: Ensemble | list[Ensemble], alpha: float = 0.15, 
            lw: int = 4, colors: list[str] = None, yrange: list[float] = None, 
            title: str = "", xlab: str = "", ylab: str = "", vline: int = -1, 
            vlab: str = "", sd: bool = False, 
            legend: bool = True
            ) -> Callable[[plt.Axes], None]:
        """
        Plot a function a(and its SD).
        """
        if isinstance(data, Ensemble): data = [data]
        if colors is not None or len(colors) < len(data): colors = self.colors
    
        def plot_fn(ax: plt.Axes) -> None:
            for d, color in zip(data, colors):
                x = np.arange(len(d.ensAvg))
                ax.plot(x, d.ensAvg+dy, color=color, lw=lw, label=d.label)
                
                if sd:
                    if len(d.sds) > 0:
                       ax.fill_between(
                           x, d.ensAvg-d.sds, d.ensAvg+d.sds, color=color, 
                           alpha=alpha
                           ) 
                
            if vline > 0:
                ax.axvline(
                    vline, color="red", lw=lw*2, linestyle=":", label=vlab
                    )
                
            ax.set_title(title, fontsize=self.fontsize)
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.set_xlim(x[0], x[-1])
            if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
            if legend: ax.legend(fontsize=self.fontsize)
            
        return plot_fn

    def plotFigs(
            self, plot_fn: Callable[[plt.Axes], None] | 
            list[Callable[[plt.Axes], None]], tight: bool = False,
            figsize: list[int] = None, vert: bool = False, title: str = ""
            ) -> None:
        """
        Generate plots using a (list of) plt.Axes object(s).
        Currently only usable for 1xn or nx1 sized figures.
        """
        if not isinstance(plot_fn, list): plot_fn = [plot_fn]
        if figsize is None: 
            figsize = self.figsize
            if vert: figsize = (figsize[0]*len(plot_fn), figsize[1])
        
        fig = plt.figure(figsize=figsize)
        if vert: subfigs = fig.subfigures(len(plot_fn), 1)
        else: subfigs = fig.subfigures(1, len(plot_fn))
        if not isinstance(subfigs, np.ndarray): subfigs = [subfigs]
        
        for subfig, fn in zip(subfigs, plot_fn):
            ax = subfig.subplots()
            fn(ax)
            
        fig.suptitle(title, fontsize=self.titleSize)
        if tight: fig.tight_layout()
        plt.show()
        
    def quickPlot(
            self, data: np.ndarray | list[np.ndarray], vert: bool = False,
            figsize: tuple[int] = None, color: str = "darkblue", lw: int = 4,
            vline: int = -1, title: str = "", xlab: str = "", ylab: str = ""
            ) -> None:
        """
        Quickly plot signal data.
        """
        if isinstance(data, np.ndarray): data = [data]
        if figsize is None: 
            figsize = self.figsize
            if vert: figsize = (figsize[0], figsize[1]*len(data))
        if vert:
            fig, axs = plt.subplots(nrows=len(data), ncols=1, figsize=figsize)
        else:
            fig, axs = plt.subplots(nrows=1, ncols=len(data), figsize=figsize)
        if isinstance(axs, plt.Axes): axs = [axs]
        
        for d, ax in zip(data, axs):
            x = np.arange(len(d))
            ax.plot(x, d, color=color, lw=lw)
            if vline > 0:
                ax.axvline(vline, color="red", lw=lw*2, linestyle=":")
            ax.set_title(title, fontsize=self.fontsize)
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.set_xlim(x[0], x[-1])
            
    