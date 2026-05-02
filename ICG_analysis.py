""" Latest version of individual and ensemble avergage analysis """
import numpy as np
import pandas as pd
from typing import Callable

from DataFormats import Ensemble, Point, Line, FindPoints
from ICGPlot import ICGPlot
        
class ICG():
    def __init__(self) -> None:
        self.fs = 1000.0 # Sampling frequency
        
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
        ecg = np.array([])
        if isinstance(features, pd.DataFrame):
            ecg = self.extractFeatures(features, ecg=True)
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
                if fBounds > rBounds*boundsMax or fBounds < rBounds*boundsMin:
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
            Ensemble(features=features, label="Kept signals", ecg=ecg),
            Ensemble(features=np.array(rem), label="Removed signals")
            ]
    
    def extractFeatures(
            self, df: pd.DataFrame, ecg: bool = False
            ) -> np.ndarray:
        """
        Extract the feature vectors out of a dataframe.
        """
        dataCols = [col for col in df.columns if col.startswith("s")]
        if ecg: features = df[dataCols].to_numpy()[-1]
        else: features = df[dataCols].to_numpy()[:-1] # ECG stored in last row
        
        return features
    
    # Runs
    def runPoints(self, data: pd.DataFrame, title: str = "") -> None:
        ens = Ensemble(
            features=icg.extractFeatures(data), 
            ecg=icg.extractFeatures(data, ecg=True)
            )
        ecg = Ensemble(ecg=icg.extractFeatures(data, ecg=True))
        
        # Find all relevant points
        f.findPoints(ens)
        ecg.r = ens.r
        ecg.t = ens.t
        
        # Plot signal and points
        ...
    
    def runBounds(
            self, data: pd.DataFrame, bounds: list[float], title: str = ""
            ) -> None:
        """
        Run the bounding filter at different bound sizes.
        """
        fns1 = []
        fns2 = []
        fns3 = []
        ensCompare = []
        for b in bounds:
            d = data.copy()
            ens, rem = self.filterByBounds(features=d, boundsMax=b)
            title=f"Bound size = {b} -> {rem.cnt}/{len(d)} complexes removed."
            fn1 = self.multiPlotShadeFn(
                data=[ens, rem], showSD=True, vline=255, vlab="R-peak",
                title=title, xlab="t [ms]", ylab=r"$\frac{dZ}{dt}$ [Ω/s]"
                )
            fn2 = self.multiPlotShadeFn(
                data=[ens, rem], showSD=True, vline=255, vlab="R-peak",
                title=title, xlab="t [ms]", ylab=r"$\frac{dZ}{dt}$ [Ω/s]",
                yrange=[-0.005, 0.014]
                )
            fns1.append(fn1)
            fns2.append(fn2)
            ensCompare.append(ens)
         
        # Plot ensemble averages with shaded SDs
        figTitle="Kept/discarded signals at different bound sizes"
        self.plotFigs(
            plot_fn=fns1, title=figTitle, vert=True
            )
        self.plotFigs(
            plot_fn=fns2, title=figTitle+" (static y-range)", vert=True
            )
        
        # Plot ensemble averages at different filtering bounds sizes
        for ens, b in zip(ensCompare, bounds):
            ens.label = str(b)
        fns3.append(
            self.multiPlotLinesFn(
                data=ensCompare, xlab="t [ms]", ylab=r"$\frac{dZ}{dt}$ [Ω/s]",
                title="Ensemble average at different bounds sizes"
            )
        )
        
        # Zoom on on relevant area
        fns3.append(
            self.multiPlotLinesFn(
                data=ensCompare, xlab="t [ms]", ylab=r"$\frac{dZ}{dt}$ [Ω/s]",
                zoomRange=[255, 500],
                title="Ensemble average at different bounds sizes (zoomed)"
            )
        )
        
        # Add small y deviation to pull signals apart
        fns3.append(
            self.multiPlotLinesFn(
                data=ensCompare, xlab="t [ms]", ylab=r"$\frac{dZ}{dt}$ [Ω/s]",
                zoomRange=[255, 500], dy = 0.0001,
                title="Ensemble average at different bounds sizes (zoomed + y delta)"
            )
        )
        
        self.plotFigs(
            plot_fn=fns3, vert=True, 
            title="Ensemble average at different bound sizes"
            )
    
icg = ICG()
plt = ICGPlot()
f = FindPoints()

df = icg.importCSV("Data/baseline_climbing_wECG.csv")
signalBounds = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
ens = Ensemble(
    features=icg.extractFeatures(df),
    sigAlt=icg.extractFeatures(df, ecg=True),
    label="test",
    )
slope = 3e-5

f.findPoints(ens)




















