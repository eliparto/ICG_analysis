""" Analysis of CIG complexes in singular and ensemble forms """

import numpy as np
import numpy.typing as npt
from typing import Callable
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.cluster.vq import vq, kmeans, whiten
import json

from icg import ICG

class Ens():
    def __init__(self, fs: int = 1000) -> None:
        self.fs = fs # Sampling frequency
        
        # Plotting params
        self.figsize = (80, 15)
        self.shape_plot = (60, 15)
        self.height_hist = 40
        self.width = 10
        self.fontsize = 30
     
    ## Runs
    def kGrid(self, dz: npt.NDArray[np.float_], dz_raw: npt.NDArray[np.float_], 
              dzdt: npt.NDArray[np.float_], idxs: list[int],
              offsetFullL: int, offsetFullR: int, 
              offsetZoomL: int, offsetZoomR: int, idx_start: int, idx_end: int, 
              kStart: int = 2, kEnd: int = 5, title: str = "",
              plot: bool = True, out: bool = False) -> None:
        """
        Initial exploratory analysis of clustering with different k-values.
        
        PARAMS:
            dzdt: Input signal of the derivative of Z0
            idxs: R-peak indices
            offset*: Space left/right of R-peak of signals to store
            k*: Min and max k-values to test sequentially
        """    
        ## Feature setup
        features_full_dzdt = self.ensemble(dzdt, idxs, 
                                           offsetFullL, offsetFullR)
        features_zoom_dzdt = self.ensemble(dzdt, idxs, 
                                           offsetZoomL, offsetZoomR)
        features_full_dzdt_raw = self.ensemble(dzdt_raw, idxs, 
                                           offsetFullL, offsetFullR)
        features_zoom_dzdt_raw = self.ensemble(dzdt_raw, idxs, 
                                           offsetZoomL, offsetZoomR)
        features_full_dz = self.ensemble(dz, idxs, offsetFullL, offsetFullR)
        features_full_dz_raw = self.ensemble(dz_raw, idxs, 
                                             offsetFullL, offsetFullR)
        
        breath = self.bandpass_filter(dz, 0.1, 0.4, self.fs)
        
        ## kMeans
        features = self.centerData(np.copy(features_zoom_dzdt)) # Input: zoomed in on complexes anchored on R-peaks and anchored in y-axis at 0
        for k in range(kStart, kEnd+1):
            centroids, labels, distances = self.kCluster(features, k)
            
            allFeatures = [
                features, features_full_dzdt,
                features_zoom_dzdt_raw, features_full_dzdt_raw,
                features_full_dz, features_full_dz_raw
                ]
            
            # Generate clustered complexes per feature set
            allComplexes = []
            for f in allFeatures: 
                allComplexes.append(self.getComplexes(f, labels))
            allComplexes_zoom, allComplexes_full, allComplexes_zoom_raw, allComplexes_full_raw, allComplexes_dz, allComplexes_dz_raw = allComplexes
            
            # Generate standard deviations per feature set
            allSds = []
            for i, complexes in enumerate(allComplexes):
                allSds.append([np.std(c, axis=0) for c in complexes])
            sds_zoom, sds_full, sds_zoom_raw, sds_full_raw, sds_dz, sds_dz_raw = allSds
            
            allPoints = self.findTimes(k, labels, idxs)
            
            # Generate ensemble average signals per feature set
            ensAvg_full = self.ensAvg(allComplexes_full)
            ensAvg_zoom_raw = self.ensAvg(allComplexes_zoom_raw)
            ensAvg_full_raw = self.ensAvg(allComplexes_full_raw)
            ensAvg_dz = self.ensAvg(allComplexes_dz)
            ensAvg_dz_raw = self.ensAvg(allComplexes_dz_raw)
            ensAvgs = [
                centroids, ensAvg_full, ensAvg_zoom_raw, ensAvg_full_raw,
                ensAvg_dz, ensAvg_dz_raw
                ]
                    
            ## Plotting
            if plot:
                clusterNames = [f"Cluster {cluster}" for cluster in range(k+1)]
                # Histogram
                hist = self.histFn(labels, title="Cluster contents", 
                                   density=False, xlab="Cluster", ylab="Count",
                                   color="skyblue")
                # Anchored zoomed-in complexes
                plot = self.plotMultFn(data=centroids, sds=sds_zoom, 
                                       colors=COLORS,
                                       title="Clustered ensemble averages",
                                       xlab="t [ms]", ylab="dz/dt [Ω/s]",
                                       labels=clusterNames, vline=50)
                # Boxplots
                bplot_dzdt = self.pointsBoxplot(dzdt, allPoints, 
                                                title="Dist of DZDT",
                                                colors=COLORS, 
                                                labels=clusterNames,
                                                xlab="Cluster", ylab="[Ω/s]")
                
                bplot_dz = self.pointsBoxplot(dz, allPoints, 
                                                title="Dist of filtered Z0",
                                                colors=COLORS, 
                                                labels=clusterNames,
                                                xlab="Cluster", ylab="[Ω]")
                
                bplot_dz_raw = self.pointsBoxplot(dz_raw, allPoints, 
                                                title="Dist of Z0",
                                                colors=COLORS, 
                                                labels=clusterNames,
                                                xlab="Cluster", ylab="[Ω]")
                
                # Comparisons of filtered and unfiltered signals
                plot_dzdt_full = self.plotMultFn(ensAvg_full, sds=sds_full,
                                                 colors=COLORS,
                                                 title="Clustered ensemble averages (full filtered signal)",
                                                 xlab="t [ms]", ylab="dz/dt [Ω/s]",
                                                 labels=clusterNames, vline=256)
                plot_dzdt_zoom_raw = self.plotMultFn(ensAvg_zoom_raw, sds=sds_zoom_raw,
                                                 colors=COLORS,
                                                 title="Clustered ensemble averages (zoomed raw signal)",
                                                 xlab="t [ms]", ylab="dz/dt [Ω/s]",
                                                 labels=clusterNames, vline=50)
                plot_dzdt_full_raw = self.plotMultFn(ensAvg_full_raw, sds=sds_full_raw,
                                                 colors=COLORS,
                                                 title="Clustered ensemble averages (full raw signal)",
                                                 xlab="t [ms]", ylab="dz/dt [Ω/s]",
                                                 labels=clusterNames, vline=256)
                
                plot_dz = self.plotMultFn(ensAvg_dz, sds=sds_dz, colors=COLORS,
                                          title="Clustered ensemble averages Z (full filtered signal)",
                                          xlab="t [ms]", ylab="z/dt [Ω]", 
                                          labels=clusterNames, vline=256)
                
                plot_dz_raw = self.plotMultFn(ensAvg_dz_raw, sds=sds_dz_raw, colors=COLORS,
                                          title="Clustered ensemble averages Z (full raw signal)",
                                          xlab="t [ms]", ylab="z/dt [Ω]", 
                                          labels=clusterNames, vline=256,
                                          ignoreSDylim=True
                                          )
                
                # Timeline plots
                t_start = 26000
                t_duration = 10000
                
                dzdtTimePlots = []
                for i in range(10):
                    timePlot = self.plotTimeLine(data=dzdt, idxs=idxs,
                                                 idx_start=t_start,
                                                 idx_end=t_start+t_duration, 
                                                 allPoints=allPoints, s=750, 
                                                 colors=COLORS,
                                                 markers=MARKERS, 
                                                 xlab="t [ms]", ylab="z [Ω]", 
                                                 title="DZDT")
                    dzdtTimePlots.append(timePlot)
                    t_start += t_duration
                
                dzTimePlots = []
                t_start=26000
                for i in range(10):
                    timePlot = self.plotTimeLine(data=dz, idxs=idxs, 
                                                 ref=breath,
                                                 idx_start=t_start,
                                                 idx_end=t_start+t_duration, 
                                                 allPoints=allPoints, s=750, 
                                                 colors=COLORS,
                                                 markers=MARKERS, 
                                                 xlab="t [ms]", ylab="z [Ω]", 
                                                 title="DZ (filtered)")
                    dzTimePlots.append(timePlot)
                    t_start += t_duration
                    
                dzRawTimePlots = []
                t_start=26000
                for i in range(10):
                    timePlot = self.plotTimeLine(data=dz_raw, idxs=idxs,
                                                 idx_start=t_start,
                                                 idx_end=t_start+t_duration, 
                                                 allPoints=allPoints, s=750, 
                                                 colors=COLORS,
                                                 markers=MARKERS, 
                                                 xlab="t [ms]", ylab="z [Ω]", 
                                                 title="DZ (raw)")
                    dzRawTimePlots.append(timePlot)
                    t_start += t_duration
                
                ## Combined plots
                # Anchored complexes w/ hist
                self.subfig(plot_fns=[plot, hist],
                            title=f"k = {k}; {len(features)} complexes -> " + title,
                            vert=True,
                            figsize=(80, 60),
                            )
                # Zoomed dzdt filtered vs raw
                self.subfig(plot_fns=[plot, plot_dzdt_zoom_raw],
                            title="DZDT filtered + anchored vs raw signal (zoom)",
                            vert=True,
                            figsize=(80, 60)
                            )
                
                # dzdt filtered zoom vs full
                self.subfig(plot_fns=[plot_dzdt_zoom_raw, plot_dzdt_full],
                            title="DZDT filtered signal (zoom vs full)",
                            vert=True,
                            figsize=(80, 60)
                            )
                
                # Full Z filtered vs raw
                self.subfig(plot_fns=[plot_dz, plot_dz_raw], 
                            title="Z0 filtered vs raw",
                            vert=True,
                            figsize=(80, 60)
                            )
                
                # Timelines
                self.subfig(plot_fns=dzdtTimePlots,
                             title=f"k = {k}: Cluster type occurences in DZDT",
                             figsize=(80, 120), vert=True)
                
                self.subfig(plot_fns=dzTimePlots,
                             title=f"k = {k}: Cluster type occurences in Z0",
                             figsize=(80, 120), vert=True)
                
                self.subfig(plot_fns=dzRawTimePlots,
                             title=f"k = {k}: Cluster type occurences in Z0 (raw)",
                             figsize=(80, 120), vert=True)
                
                # Boxplots
                self.subfig(plot_fns=[bplot_dzdt, bplot_dz, bplot_dz_raw],
                            title="Distributions of DZDT/Z0 values",
                            figsize=(80, 40))
                
                # self.subfig(plot_fns=timePlot,
                #             title="Cluster type occurences")
                
        if out: return ensAvgs, allComplexes, allSds, labels
            
    ## Feature setup
    def ensemble(self, data: npt.NDArray[np.float_], idxs: list[int], 
            offsetL: int = 50, offsetR: int = 200) -> npt.NDArray[np.float_]:
        """
        Collect ensembles with custom left/right offsets.
        """
        ensembles = np.array([data[i-offsetL:i+offsetR] for i in idxs])
        return ensembles
    
    def ensAvg(self, data: list[npt.NDArray[np.float_]]
               ) -> npt.NDArray[np.float_]:
        """
        Generate ensemble average signals based on an ensemble of signals.
        """
        avgs = np.array([np.average(d, axis=0) for d in data])
        return avgs
    
    def getComplexes(self, data: npt.NDArray[np.float_], 
                     labels: npt.NDArray[np.float_]
                     ) -> list[npt.NDArray[np.float_]]:
        """
        Generate a list of complexes based on cluster labels.
        """
        assert len(data) == len(labels), f"Input data mismatch ({len(data)} and {len(labels)})"
        allComplexes = []
        for clusterIdx in range(len(np.unique(labels))):
            complexes = np.array([
                data[i] for i in range(len(labels)) if labels[i] == clusterIdx
                ])
            allComplexes.append(complexes)
            
        return allComplexes        
    
    def centerData(self, data: npt.NDArray[np.float_]
                   ) -> npt.NDArray[np.float_]:
        """
        Alter vectors to start and end with a y-value of 0.
        """
        for i, d in enumerate(data):
            data[i] = d + np.linspace(-d[0], -d[-1], len(d))
            
        return data        
    
    def lowpass_filter(self, sig: npt.NDArray[np.float_], cutoff_hz, fs_hz, 
                       order=4) -> npt.NDArray[np.float_]:
        """
        Butterworth lowpass filter
        """
        nyq = fs_hz / 2
        b, a = butter(order, cutoff_hz / nyq, btype='low')
        return filtfilt(b, a, sig) 

    def bandpass_filter(self, sig: npt.NDArray[np.float_], low_hz: float, 
                        high_hz: float, fs_hz: float, order: int = 2
                        ) -> npt.NDArray[np.float_]:
        """
        Butterworth bandpass filter
        """
        nyq = fs_hz / 2
        b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
        return filtfilt(b, a, sig)
    
    ##  Clustering
    def kCluster(self, features: npt.NDArray[np.float_], k: int
                 ) -> list[npt.NDArray[np.float_]]:
        """
        Perform k-means clustering: whitening -> clustering.
        """
        sds = np.std(features, axis=0) # SD to reconstruct vectors
        features_w = whiten(features) # Whitened features
        centroids, distortion = kmeans(features_w, k)
        
        labels, distances = vq(features_w, centroids) # Cluster ids and dist per cluster
        centroids_scaled = centroids * sds # Recover averaged cluster signals
        
        return centroids_scaled, labels, distances
    
    def findTimes(self, k, labels: npt.NDArray[np.int_], idxs: list[int]
                  ) -> list[list[int]]:
        """ 
        Cross-fereference the times of occurance of clustered complexes.
        """
        allPoints = []
        for kIdx in range(k):
            points = [idxs[i] for i in range(len(labels)) if labels[i] == kIdx]
            allPoints.append(points)
            
        return allPoints
    
    ## Plotting
    def pointsBoxplot(self, data: npt.NDArray[np.float_], allPoints: list[int], 
                      title: str = "", colors: list[str] | str = "skyblue",
                      labels: list[str] = None, xlab: str = "", 
                      ylab: str = None 
                      ) -> Callable[[plt.Axes], None]:
        """
        Generate a boxplots of y-axis positions.
        """
        allYPoints = []
        for i, xPoints in enumerate(allPoints):
            xPoints = np.array(xPoints)
            yPoints = data[xPoints]
            allYPoints.append(yPoints)
                        
        if isinstance(colors, list): colors = colors[:len(allPoints)]
        else: colors = [colors for _ in range(len(allPoints))]
        # if isinstance(labels, list): labels=labels[:-1]
        
        labels = [str(i) for i in range(len(allPoints))]
        
        def plot_fn(ax: plt.Axes) -> None:
            bplot = ax.boxplot(allYPoints, patch_artist=True, 
                       labels=labels)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.set_title(title, fontsize=self.fontsize)
            
        return plot_fn
    
    def plotTimeLine(self, data: npt.NDArray[np.float_], idxs: list[int],
                     idx_start: int, idx_end: int, allPoints: list[list[int]],
                     ref: npt.NDArray[np.float_] = None,
                     title: str = "", xlab: str = "", ylab: str = "", 
                     colors: list[str] | str = "skyblue", 
                     markers: list[str] | str = ".", lw: int = 4, s=10
                     ) -> Callable[[plt.Axes], None]:
        """
        Plot a signal over time with some points placed over it.
        """
        x = np.arange(idx_start, idx_end, 1)
        if isinstance(colors, str): 
            colors = [colors for _ in range(len(allPoints))]
        if isinstance(markers, str): 
            markers = [markers for _ in range(len(allPoints))]
        
        def plot_fn(ax: plt.Axes) -> None:
            ax.plot(x, data[x], color="skyblue", lw=lw, zorder=1)
            if isinstance(ref, np.ndarray):
                ax.plot(x, ref[x], color="pink", lw=lw, zorder=2, alpha=0.6,
                        label="Breath")
            
            for i, xPoints in enumerate(allPoints):
                xPoints = np.array(xPoints)
                yPoints = data[xPoints]
                ax.scatter(xPoints, yPoints, s=s, color=colors[i], 
                           marker=markers[i], label=f"Cluster {i}",
                           zorder=(i+1)*10)
            
            ax.set_xlim(x[0], x[-1])
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.legend(fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.set_title(title, fontsize=self.fontsize)
        
        return plot_fn
    
    def plotFn(self, data: npt.NDArray[np.float_], 
               sd: npt.NDArray[np.float_] = None, title: str = "", 
               color: str = "skyblue", alpha: float = 0.3, xlab: str = "", 
               ylab: str = "", lw: int = 4) -> Callable[[plt.Axes], None]:
        """
        Plot a single signal with or without its standard deviation.
        """
        assert data.ndim == 1, f"{data.ndim}-D data passed to 1-D plot."
        x = np.arange(0, len(data), 1)
        
        def plot_fn(ax: plt.Axes) -> None:
            ax.plot(x, data, color=color, lw=lw)
            if isinstance(sd, np.ndarray): 
                ax.fill_between(x, data+sd, data-sd, color=color, alpha=alpha)
            ax.set_xlim(x[0], x[-1])
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.set_title(title, fontsize=self.fontsize)
            
        return plot_fn
    
    def plotMultFn(self, data: list[npt.NDArray[np.float_]], 
                   ref: npt.NDArray[np.float_] = None,
                   sd: npt.NDArray[np.float_] = None,
                   sds: list[npt.NDArray[np.float_]] = None, vline: int = -1,
                   refColor: str = "blue", colors: str | list[str] = "skyblue", 
                   title: str = "", labels: list[str] = None, xlab: str = "", 
                   ylab: str = "", alpha: float = 0.15, lw: int = 4,
                   ignoreSDylim: bool = False
                   ) -> Callable[[plt.Axes], None]:
        """
        Plot multiple curves with or without their standard deviations.
        ref: reference signal such as an ensemble average
        data: multiple signals to plot sequentially
        """
        x = np.arange(0, len(data[0]), 1)
        if isinstance(colors, str): colors = [colors for _ in range(len(data))]
        if isinstance(colors, list): 
            if len(colors) < len(data): 
                colors = ["skyblue" for _ in range(len(data))]
        if not isinstance(labels, list): 
            labels = ["" for _ in range(len(data))]
        
        def plot_fn(ax: plt.Axes) -> None:
            for i, d in enumerate(data):
                ax.plot(x, d, color=colors[i], lw=lw, label=labels[i])
                auto_ylim = ax.get_ylim()
                ax.autoscale(enable=True, axis='y', tight=False)
                
            if isinstance(sds, list):
                for i, d in enumerate(data):
                    ax.fill_between(
                        x, d+sds[i], d-sds[i], color=colors[i], alpha=alpha
                        )
                
            if isinstance(ref, np.ndarray): 
                ax.plot(x, ref, color=refColor, lw=lw, label="Reference")
                if isinstance(sd, np.ndarray): ax.fill_between(
                        x, ref+sd, ref-sd, color=refColor, alpha=alpha
                        )
            
            if "" not in labels: ax.legend(fontsize=self.fontsize)  
            if vline > 0: 
                ax.axvline(vline, color="darkgrey", lw=lw*2, linestyle=":",
                           label="Anchor point")            
            ax.set_xlim(x[0], x[-1])
            ax.grid(which="both", axis="both")
            ax.tick_params(axis="both", labelsize=self.fontsize)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)
            if ignoreSDylim: 
                dYlim = auto_ylim[1] - auto_ylim[0]
                auto_ylim = (
                    auto_ylim[0] - 0.2*dYlim, auto_ylim[1] + 0.2*dYlim
                    )
                ax.set_ylim(auto_ylim)
            ax.set_title(title, fontsize=self.fontsize)
            
        return plot_fn
    
    def heatmapFn(self, data: npt.NDArray[np.float_], cmap: str = "magma", 
                  xlab: str = "", ylab: str = "", drawLine: int = -1, 
                  title: str = None) -> Callable[[plt.Axes], None]:
        """
        Generate a signal heatmap.
        """
        assert data.ndim == 2, f"{data.ndim}-D data passed to 2-D plot."
        
        def plot_fn(ax: plt.Axes) -> None:
            ax.imshow(data, aspect="auto", cmap=cmap)
            if drawLine > 0: ax.axvline(x=drawLine, color="red", linewidth=3, 
                       label="")
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.set_title(title, fontsize=self.fontsize)
            
        return plot_fn
        
    def histFn(self, data: npt.NDArray[np.int_], density: bool = False, 
               title: str = "", xlab: str = "", ylab: str = "", 
               color: str = "skyblue", showCnt: bool = False
               ) -> Callable[[plt.Axes], None]:
        """
        Generate a histogram of counts or probability densities.
        """
        assert data.ndim == 1, f"{data.ndim}-D data passed. 1-D needed."
        
        def plot_fn(ax: plt.Axes) -> None:
            indivVals = np.unique(data)
            bins = np.arange(indivVals.min() - 0.5, indivVals.max() + 1.5)
            _, _, patches = ax.hist(data, bins=bins, density=density, 
                                    color=color)
            
            ax.set_title(title, fontsize=self.fontsize)
            ax.set_xticks(indivVals)
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)
            ax.tick_params(axis="both", labelsize=self.fontsize)

            
        return plot_fn
        
    def subfig(self, plot_fns: Callable[[plt.Axes], None] | list[Callable[[plt.Axes], None]], 
               figsize: tuple[int] = None, vert: bool = False, 
               title: str = "") -> None:
        """
        Define a subfigure layout.
        """
        if not isinstance(plot_fns, list): plot_fns = [plot_fns]
        if figsize == None: figsize = self.figsize
        fig = plt.figure(figsize=figsize)
        if (vert): subfigs = fig.subfigures(len(plot_fns), 1)
        else: subfigs = fig.subfigures(1, len(plot_fns))
        
        if (len(plot_fns) == 1): subfigs = [subfigs]
        for subfig, plot_fn in zip(subfigs, plot_fns):
            ax = subfig.subplots()
            plot_fn(ax)
            
        fig.suptitle(title, fontsize=self.fontsize*2)
        plt.show()     
        
    ## Data importing
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
    
COLORS = [
    "#E74C3C",  # red
    "#2ECC71",  # green
    "#E67E22",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#F1C40F",  # yellow
    "#E91E63",  # pink
    "#FF5722",  # deep orange
]

MARKERS = [
    "o",   # circle
    "s",   # square
    "^",   # triangle up
    "D",   # diamond
    "P",   # plus (filled)
    "X",   # x (filled)
]

ens = Ens()    
icg = ICG()

times = icg.importJson("data/CLIMBING_ELI/peaks_1.json")
ticks = icg.importBin("Data/CLIMBING_ELI/TicksA.bin")
ecg = icg.importBin("Data/CLIMBING_ELI/FILTECG.bin")
dzdt = icg.importBin("Data/CLIMBING_ELI/FILTDZDT.bin") # First derivative of ICG
dzdt_raw = icg.importBin("Data/CLIMBING_ELI/DZDT.bin") 
dz = icg.importBin("Data/CLIMBING_ELI/DZ.bin") # ICG data
dz_raw = icg.importBin("Data/CLIMBING_ELI/Z0.bin")
idxs = icg.findIdxs(ticks, times)

# Inputs
features_full = ens.ensemble(dzdt, idxs, offsetL=256, offsetR=744)
features_zoom = ens.ensemble(dzdt, idxs)
features_zoom_centered = ens.centerData(ens.ensemble(dzdt, idxs))

# Runs
# ens.kGrid(features_full, idxs, title="(Full complexes)")
# ens.kGrid(features_zoom, idxs, title="(Partial complexes)")
a = ens.kGrid(dz=dz, dz_raw=dz_raw, dzdt=dzdt, idxs=idxs, offsetFullL=256,
          offsetFullR=744, offsetZoomL=50, offsetZoomR=200, idx_start=26000,
          idx_end=46000, kStart=2, kEnd=6)



















