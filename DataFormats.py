import numpy as np
from dataclasses import dataclass, field

@dataclass
class Point:
    """
    Simple point holding a time and value.
    """
    x: int = None
    y: float = None

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
class Ensemble: 
    """
    An ensemble (collection of features/signals) and related data.
    """
    features: np.ndarray = field(default_factory=lambda: np.array([])) # 2D
    ensAvg: np.ndarray = field(default_factory=lambda: np.array([])) # 1D
    sds: np.ndarray = field(default_factory=lambda: np.array([])) # 1D
    ecg: np.ndarray = field(default_factory=lambda: np.array([])) # 1D
    label: str = ""
    cnt: int = -1
    r: Point = None
    t: Point = None
    b: Point = None
    c: Point = None
    x: Point = None
    bVals: dict = field(default_factory=lambda: {})
    cVals: dict = field(default_factory=lambda: {})
    xVals: dict = field(default_factory=lambda: {})
    
    gen_avg = staticmethod(lambda l: np.average(l, axis=0)
                           if len(l) > 0 and l.ndim == 2 else np.array([]))
    gen_sds = staticmethod(lambda l: np.std(l, axis=0)
                           if len(l) > 0 and l.ndim == 2 else np.array([]))
    
    def __post_init__(self) -> None:
        self.calc()
    
    def calc(self) -> None:
        self.cnt = len(self.features)
        if len(self.features) > 0:
            self.ensAvg = self.gen_avg(self.features)
            self.sds = self.gen_sds(self.features)
            
class FindPoints():
    def __init__(self) -> None:
        ...
    
    def findB_lozano(self, signal: Ensemble) -> Ensemble:
        """
        Find the B-point according to Lozano (max perpendicular dist from cord
        spanning from R-peak projection to C point by looking for the same 
        slope on the curve.
        """
        def calcDist(pLine: Point, pDZ: Point, slope: float) -> float:
            """
            Calculate the length of the perpendicular line between two points.
            pLine: point on the line from the R-peak projection to C point
            pDZ: point on DZDT
            """
            line_RC = Line(a=slope, pt=pLine)
            line_norm = Line(a=-slope, pt=pDZ)
            pI = line_RC.calcIntersection(line_norm) # Intersection
            
            return np.sqrt((pI.x - pLine.x)**2 + (pI.y - pLine.y)**2)
            
        # Find the slope of the projection of t_R on DZDT to C
        dtSlope = signal.c - signal.r
        dzSlope = signal.ensAvg[signal.c] - signal.ensAvg[signal.r]
        slope = dzSlope / dtSlope
        
        dAavg = np.gradient(signal.ensAvg)
        
    
    def findB_derivs(self, signal: Ensemble) -> None:
        """
        Find the B-points corresponding to the peaks of the third and fourth
        derivatives of Z0.
        """
        ...
        
    def findB_inflecion(self, signal: Ensemble) -> None:
        """
        Find the B-point by looking for the inflection point of the upward slope.
        """
        ...
    
    def findT(self, signal: Ensemble) -> None:
        """
        Find the ECG T-point by looking for the first peak after
        the (last) C-point.
        """
        tStart = next(reversed(signal.cVals.values())).t
        sig = np.copy(signal.ensAvg)[tStart]
        dSig = np.gradient(sig)
        
        # Find the first peak (zero crossing)
        t = np.where(np.diff(np.sign(dSig)))[0][0] + tStart
        signal.t = Point(x=t, y=signal.ensAvg[t])
    
    def findC(
            self, signal: Ensemble, tStart: int = 255, duration: int = 200
            ) -> None:
        """
        Find the C-point(s) of an ensemble average signal.
        Finds the zero-crossings of the relevant subset of the data.
        """
        sig = np.copy(signal.ensAvg)[tStart:tStart+duration]
        dSig = np.gradient(sig)
        
        # Find the zero scrossing(s) -> Choose second peak if 1.4x greater
        pts = np.where(np.diff(np.sign(dSig)))[0] + tStart
        cVals = {}
        cVals["pk1"] = Point(
            x=pts[0], y=signal.ensAvg[pts[0]]
            )
        signal.c = cVals["pk1"]
        if (len(pts) >= 3): 
            cVals["pk2"] = Point(
                x=pts[2], y=signal.ensAvg[pts[2]]
                )
            if cVals["pk2"].z > cVals["pk1"].z * 1.4: signal.c = cVals["pk2"]
            
        signal.cVals = cVals

    def findEqualSlopes(self, slope: float, sig: np.ndarray) -> list[int]:
        """
        Find time-point(s) in DZ with similar slope as the reference slope:
        Look for minima in |d(sig)/dt - slope|
        """
        d_diff = np.abs(np.gradient(sig) - slope)
        
        
        
        
        
        
        
        
        
        
        
        
        
        