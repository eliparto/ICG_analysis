import numpy as np
from matplotlib import pyplot as plt
from typing import Callable

from DataFormats import Ensemble, Point

class ICGPlot():
    def __init__(self) -> None:
        self.figsize = (80,20)
        self.fontsize = 40
        self.titlesize = 60
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
        
        # Marker info/properties
        self.markers = {
            "lozano":     {"lbl": "Lozano",     "pos": "left",   "show": True},
            "inflection": {"lbl": "Inflect",    "pos": "left",   "show": True},
            "d2":         {"lbl": r"$d^3$",     "pos": "right",  "show": True},
            "d3":         {"lbl": r"$d^4$",     "pos": "right",  "show": True},
            "c":          {"lbl": "$C$",        "pos": "top",    "show": True},
            "c_1":        {"lbl": r"$C_1$",     "pos": "top",    "show": True},
            "c_2":        {"lbl": r"$C_2$",     "pos": "top",    "show": True},
            "t":          {"lbl": r"$T$",       "pos": "top",    "show": False},
            "r":          {"lbl": r"$R$",       "pos": "bottom", "show": True},
        }
     
    # Plotting functions
    def plotUnderlayFn(
            self, data: Ensemble, nSignals: int, alpha: float = 0.15, 
            lwMain: int = 6, lw: int = 4, color: str = "darkblue", 
            yrange: list[float] = None, vline: int = -1, vlab: str = "", 
            title: str = "", xlab: str = "", ylab: str = ""
            ) -> Callable[[plt.Axes], None]:
        """
        Plot a function with underlying lighter shaded functions.
        """
        if color is None: color = self.colors[0]
        if nSignals > len(data.features): nSignals = len(data.features)
        sigIdxs = np.random.choice(len(data.features), nSignals, replace=False)
        
        def plot_fn(ax: plt.Axes) -> None:
            x = np.arange(len(data.sig))
            ax.plot(x, data.sig, color=color, lw=lwMain)
            
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
            dy: float = 0.0, yrange: list[float] = None, vline: int = -1, 
            vlab: str = "", title: str = "", xlab: str = "", ylab: str = "",
            legend: bool = True
            ) -> Callable[[plt.Axes], None]:
        """
        Plot multiple functions with visual settings for zoom, separation etc.
        """
        if colors is None or len(colors) < len(data): colors=self.colors
        
        def plot_fn(ax: plt.Axes) -> None:
            sum_dy = 0.0
            for d, color in zip(colors):
                x = np.arange(len(d.sig))
                ax.plot(x, d.sig+dy, color=color, lw=lw, label=d.label)
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
    
    def dzdtFn(
            self, data: Ensemble, pointLabels: dict[str, bool] = None,
            showC: bool = True, showAllC: bool = True, 
            color: str = "dodgerblue", mColor: str = "darkgray", 
            shadeColor: str = "skyblue", lw: int = 4, ptSize: int = 750, 
            shadeRange: list[int] = None, yrange: list[float] = None, 
            title: str = "", xlab: str = "", ylab: str = "", 
            legend: bool = False
            ) -> Callable[[plt.Axes], None]:
        """
        Standard plotting fn for DZDT plot with relevant markera.
        """
        x = np.arange(len(data.sig))
        
        def plotFn(ax: plt.Axes) -> None:
            zorder = 10
            ax.plot(x, data.sig, color=color, lw=lw, zorder=zorder)
            
            # Plot points
            for pt in data.getAllPoints():
                zorder += 10
                self.draw_point(ax=ax, pt=pt, zorder=zorder)    
            
            ax.set_title(title, fontsize=self.fontsize)
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.set_xlim(x[0], x[-1])
            if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
            else: ax.set_ylim(self.updateYRange(ax=ax, mult=0.33)) 
            if legend: ax.legend(fontsize=self.fontsize)
            
        return plotFn
            
    def plotFn(
            self, data: Ensemble | list[Ensemble], alpha: float = 0.15, 
            lw: int = 4, colors: list[str] = None, yrange: list[float] = None, 
            title: str = "", xlab: str = "", ylab: str = "", vline: int = -1, 
            vlab: str = "", sd: bool = False, 
            legend: bool = True
            ) -> Callable[[plt.Axes], None]:
        """
        Plot a function (and its SD).
        """
        if isinstance(data, Ensemble): data = [data]
        if colors is not None or len(colors) < len(data): colors = self.colors
    
        def plot_fn(ax: plt.Axes) -> None:
            for d, color in zip(data, colors):
                x = np.arange(len(d.sig))
                ax.plot(x, d.sig, color=color, lw=lw, label=d.label)
                
                if sd:
                    if len(d.sds) > 0:
                       ax.fill_between(
                           x, d.sig-d.sds, d.sig+d.sds, color=color, 
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

    # Set up a 'canvas' and pass the plotting funcs of plots to show
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
            
        fig.suptitle(title, fontsize=self.titlesize)
        if tight: fig.tight_layout()
        plt.show()
    
    # For quick plotting
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
    
    # Graph range formatting
    def updateYRange(self, ax: plt.Axes, mult: float = 0.33) -> list[float]:
        ticks = ax.get_yticks()
        spacing = (ticks[1] - ticks[0]) * mult
        ymin, ymax = ax.get_ylim()
        
        return [ymin - spacing, ymax + spacing]        
    
    # Marker plotting helpers/wrappers
    def draw_point(
            self, ax: plt.Axes, pt: Point, zorder: int, color: str = 'black',
            offset_pts: int = 80
            ) -> None:
        marker = self.markers[pt.label]
        if not marker["show"]:
            return
        
        pos = marker["pos"]
        offsets = {
            "top":    (0,           offset_pts // 4),  # shorter for vertical
            "bottom": (0,          -offset_pts // 4),
            "right":  (offset_pts,  0),
            "left":   (-offset_pts, 0),
        }
        ha_map = {"top": "center", "bottom": "center", "right": "left",   "left": "right"}
        va_map = {"top": "bottom", "bottom": "top",    "right": "center", "left": "center"}
        
        is_vertical = pos in ("top", "bottom")
        arrow = "->" if not is_vertical else "-"
        arrow_alpha = 0.0 if is_vertical else 1.0
    
        ax.plot(pt.x, pt.y, 'o', color=color, markersize=15, zorder=zorder)
    
        ax.annotate(
            marker["lbl"],
            xy=(pt.x, pt.y),
            xytext=offsets[pos],
            textcoords='offset points',
            fontsize=self.fontsize,
            fontweight='normal',
            ha=ha_map[pos],
            va=va_map[pos],
            color=color,
            arrowprops=dict(arrowstyle=arrow, color=color, lw=2,
                            shrinkA=0, shrinkB=4, alpha=arrow_alpha),
            zorder=zorder,
        )
            
plot = ICGPlot()  