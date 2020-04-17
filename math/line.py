from itertools import product as prod
from numpy import array

class interval:
    def __init__(self, a,b):
        self.a,self.b = a,b
    
    def contains(self, x):
        return self.a<=x<=self.b

class line:
    def __init__(self, A,B):
        ''' A,B are (x,y) points'''
        self.A = A
        self.B = B
        
    def in_domain(self,x):
        Ax,_ = self.A
        Bx,_ = self.B
        return Ax <= x <= Bx
    
    def __call__(self,x):
        Ax,Ay = self.A
        Bx,By = self.B
        return Ay + ( (By-Ay) * (x-Ax)/(Bx-Ax) )
    def __str__(self):
        return (self.A, self.B)

class piecewise_line:
    def __init__(self):
        self.points = []
    
    def add(self, x,y):
        y = array(y)
        self.points.append( (x,y) ) 
        self.build()
        
    def build(self):
        self.segments = []
        P = self.points
        k = lambda a:a[0]
        sorted(P, key=k)
        for i in range(len(P)-1):
            L = line(P[i], P[i+1])
            self.segments.append(L)

    def __call__(self, x):
        S = self.segments
        for i in range(len(S)):
            if S[i].in_domain(x):
                return S[i](x)
