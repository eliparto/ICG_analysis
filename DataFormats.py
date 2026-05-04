import numpy as np
from dataclasses import dataclass, field
from scipy.signal import butter, filtfilt

@dataclass
class Point:
    """
    Simple point holding a time and value.
    """
    x: int = None
    y: float = None
    label: str = ""
    
    def getCopy(self) -> "Point":
        return Point(self.x, self.y, self.label)

@dataclass
class Line:
    """
    Holds a linear function (y = ax + b) and calculating functions.
    """
    a: float = None
    b: float = None
    pt: Point = None
        
    def __post_init__(self) -> None:
        if self.a is not None and self.pt is not None: self.calcParams()
    
    def calcParams(self, pt: Point = None, slope: float = None) -> None:
        if slope is None: slope = self.a
        if pt is None: pt = self.pt
        self.a = slope
        x = float(pt.t)
        y = pt.z
        
        self.b = y - slope * x
        
    def calcX(self, y: float) -> float:
        return (y - self.b) / self.a
        
    def calcY(self, x: float) -> float:
        return self.a * x + self.b
    
    def calcIntersection(self, line: "Line") -> "Point":
        x = self.calcX(line.point.y)
        y = self.calcY(line.point.x)
        return Point(x=x, y=y)

@dataclass
class Ensemble(): 
    """
    An ensemble (collection of features/signals) and related data.
    """
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    sig: np.ndarray = field(default_factory=lambda: np.array([])) # Main sig
    sigAlt: np.ndarray = field(default_factory=lambda: np.array([])) # Alt sig
    sds: np.ndarray = field(default_factory=lambda: np.array([])) 
    label: str = ""
    cnt: int = -1
    r: Point = None
    t: Point = None
    b: Point = None
    c: Point = None
    bPoints: dict[str, Point] = field(default_factory=lambda: {})
    cPoints: dict[str, Point] = field(default_factory=lambda: {})
    
    gen_avg = staticmethod(lambda l: np.average(l, axis=0)
                           if len(l) > 0 and l.ndim == 2 else np.array([]))
    gen_sds = staticmethod(lambda l: np.std(l, axis=0)
                           if len(l) > 0 and l.ndim == 2 else np.array([]))
    
    def __post_init__(self) -> None:
        self.calc()
    
    def calc(self) -> None:
        self.cnt = len(self.features)
        if len(self.features) > 0:
            self.sig = self.gen_avg(self.features)
            self.sds = self.gen_sds(self.features)
    
    def setC(self, peak: int = 0) -> None:
        """
        Choose the C-point:
        Specify the peak to place the C-point on (in case of multiple peaks).
            peak: auto (0), first (1) or second (2) peak
        """
        assert 0 <= peak <= 2, f"Invalid peak val {peak} -> 0 <= peak <= 2"
        cPoints = dict(self.cPoints)
        self.c = cPoints["c_1"].getCopy()
        if peak == 2: self.c = cPoints["c_2"]
        elif peak == 0 and cPoints["c_2"].y > 1.4*cPoints["c_1"].y:
            self.c = cPoints["c_2"].getCopy()
        self.c.label = "c"
    
    def clearPoints(self) -> None:
        self.r, self.t, self.b, self.c, self.x = None
        self.bPoints = {}    
        self.cPoints = {}
        self.xPoints = {}
        
    def recalcPoints(self) -> None:
        """
        Recalculate y values based on present internal signal for all points.
        """
        for pt in self.bPoints.values(): pt.y = self.sig[pt.x]
        for pt in self.cPoints.values(): pt.y = self.sig[pt.x]
        
    def getBPoints(self) -> list[Point]:
        return list(self.bPoints.values())
        
    def getCPoints(self) -> list[Point]:
        return list(self.cPoints.values())
    
    def getAllPoints(self) -> list[Point]:
        return self.getBPoints() + self.getCPoints()
            
class FindPoints():
    def __init__(self) -> None:
        self.dt = 5 # Search bound delta to eg not include min/max at bounds
    
    def findPoints(self, signal: Ensemble) -> None:
        """
        Run points detection in the correct order.
        """
        self.findR(signal)
        self.findC(signal)
        self.findT(signal)
        # self.findX(signal)
        self.findB_lozano(signal)
        self.findB_inflection(signal)
        self.findB_derivs(signal)
    
    def findB_lozano(self, signal: Ensemble) -> Ensemble:
        """
        Find the B-point according to Lozano (max perpendicular dist from cord
        spanning from R-peak projection to C point by looking for the same 
        slope on the curve.
        """
        def calcDist(ptLine: Point, ptDZ: Point, slope: float) -> float:
            """
            Calculate the length of the perpendicular line between two points.
            pLine: point on the line from the R-peak projection to C point
            pDZ: point on DZDT
            """
            line_RC = Line(a=slope, pt=ptLine)
            line_norm = Line(a=(1/slope), pt=ptDZ) # Normal vect through point
            ptI = line_RC.calcIntersection(line_norm) # Intersection point
            
            return np.sqrt((ptI.x - ptLine.x)**2 + (ptI.y - ptLine.y)**2)
        
        # Find the points on DZDT with equal slopes to the line RC and dists
        slope = (signal.c.y - signal.r.y) / (signal.c.x - signal.r.x)
        tPts = self.findEqualSlopes(signal.sig, slope, signal.r.x, signal.c.x) 
        allPts = [Point(x=t, y=signal.sig[t], label="lozano") for t in tPts]
        
        if len(tPts) == 1: signal.bPoints["lozano"] = allPts[0]
        else: # > 1 point found -> Take point with largest perpendicular dist
            dists = np.array([calcDist(signal.r, pt, slope) for pt in allPts])
            signal.bPoints["lozano"] = allPts[np.argmax(dists)]
    
    def findB_derivs(self, signal: Ensemble) -> None:
        """
        Find the B-points corresponding to the peaks of the third and fourth
        derivatives of Z0.
        """
        # d* -> * denotes order of differentiation
        d2Sig = self.filtButterLow(self.nDiff(signal.sig, n=2), f=40)
        d3Sig = np.gradient(d2Sig)
        d4Sig = np.gradient(d3Sig)
        
        # Use search bounding to be safe
        tStart = signal.r.x + self.dt
        t_d2 = np.where(np.diff(np.sign(d3Sig[tStart:])) < -1)[0][0] + tStart
        t_d3 = np.where(np.diff(np.sign(d4Sig[tStart:])) < -1)[0][0] + tStart
        
        signal.bPoints["d2"] = Point(x=t_d2, y=signal.sig[t_d2], label="d2")
        signal.bPoints["d3"] = Point(x=t_d3, y=signal.sig[t_d3], label="d3")
        
    def findB_inflection(self, signal: Ensemble) -> None:
        """
        Find the B-point by looking for the inflection point of the upward slope.
        """
        tStart = signal.r.x + self.dt
        tStop = signal.c.x - self.dt
        d2Sig = self.nDiff(signal.sig, 2)
        d2Sig = self.filtButterLow(d2Sig, f=40)
        t = np.where(
            np.diff(np.sign(d2Sig[tStart:tStop])) < 0
            )[0][0] + tStart
        
        signal.bPoints["inflection"] = Point(
            x=t, y=signal.sig[t], label="inflection"
            )
        
    def findC(
            self, signal: Ensemble, duration: int = 250
            ) -> None:
        """
        Find the C-point(s) of an ensemble average signal.
        Finds the zero-crossings of the relevant subset of the data.
        """
        # Find the zero scrossing(s)
        tStart = signal.r.x + self.dt
        dSig = np.gradient(signal.sig)[tStart:tStart+duration]
        pts = np.where(np.diff(np.sign(dSig)) < 0)[0] + tStart
        signal.cPoints["c_1"] = Point(
            x=pts[0], y=signal.sig[pts[0]], label="c_1"
            )
        
        if len(pts) > 1:
            signal.cPoints["c_2"] = Point(
                x=pts[1], y=signal.sig[pts[1]], label="c_2"
                )
            
        signal.setC()
    
    def findR(self, signal: Ensemble, t: int = 255) -> None:
        """
        Set the R-point.
        """
        signal.r = Point(x=t, y=signal.sig[t], label="r")
    
    def findT(self, signal: Ensemble, duration: int = 100) -> None:
        """
        Find the ECG T-point by looking for the first peak after
        the (last) C-point.
        """
        tStart = signal.c.x + self.dt
        sig = np.copy(signal.sigAlt)[tStart:tStart+duration]
        dSig = np.gradient(sig)
        
        # Find the first peak after the C point(s)
        t = np.argmax(dSig) + tStart
        signal.t = Point(x=t, y=signal.sig[t], label="t")
        
    def findX(self, signal: Ensemble) -> None:
        """
        Find the X-point by looking for the minima after the T-point.
        TODO: Implement
        """
        ...

    # Helper functions
    def nDiff(self, sig: np.ndarray, n: int) -> np.ndarray:
        """
        Perform n differentiation steps.
        """
        for _ in range(n): sig = np.gradient(sig)
        return sig
    
    def findEqualSlopes(
            self, sig: np.ndarray, slope: float, tStart: int, tStop: int
            ) -> list[int]:
        """
        Find time-point(s) in DZ with similar slope as the reference slope:
        Look for minima in |d(sig)/dt - slope|
        """
        dSig = np.gradient(sig)[tStart:tStop]
        return np.where(np.diff(np.sign(dSig-slope)) > 0)[0] + tStart
    
    def filtButterLow(
            self, sig: np.ndarray, f: float = 0.5, order: int = 4, 
            fs: float = 1000
            ) -> np.ndarray:
        """
        Apply a butterworth low-pass filter to a signal.
        """
        nyq = fs / 2
        b, a = butter(order, f/nyq, btype="low")
        return filtfilt(b, a, sig)
    
    def filtButterBand(
            self, sig: np.ndarray, fLow: float = 40, fHigh: float = 25,
            order: int = 4, fs: float = 1000
            ) -> np.ndarray:
        """
        Apply a butterworth band-pass filter to a signal.
        """
        nyq = fs / 2
        b, a = butter(order, [fLow/nyq, fHigh/nyq], btype="band")
        return filtfilt(b, a, sig)
    
f = FindPoints()
        
        
        
    