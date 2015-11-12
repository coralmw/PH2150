import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

class ForcePlot(Figure):
    """High level class that knows how to plot a force.
    """

    def __init__(self, *args, **kwargs):
        Figure.__init__(self, *args, **kwargs)

    def visualise(self, r_start=1, r_stop=100, n=1000):
        '''ForcePlot is a subclass of Figure to allow it to plot itself.
        '''
        r = np.linspace(r_start, r_stop, n)
        g = self.force(r)

        ax = self.gca()
        ax.plot(r, g)
        ax.set_xlabel('distance')
        ax.set_ylabel('force, N')
        self.set_title()


class InvSquForcePlot(ForcePlot):
    """Class that knows how to model a inv square force, and plot
    """

    def __init__(self, *args, **kwargs):
        self.v1 = kwargs.pop('v1')
        self.v2 = kwargs.pop('v2')
        self.const = kwargs.pop('constant')
        ForcePlot.__init__(self, *args, **kwargs)

    def force(self, r):
        return self.const*self.v1*self.v2/r**2


class GravityPlot(InvSquForcePlot):
    """Gravity plot as a special case of inv square force."""

    def __init__(self, M, m, *args, **kwargs):
        self.M, self.m = M, m
        G = 6.6748e-11
        InvSquForcePlot.__init__(self, *args, v1=M, v2=m, constant=G, **kwargs)

    def set_title(self):
        self.gca().set_title("Gravity, M={}, m={}".format(self.M, self.m))


class ElectricPlot(InvSquForcePlot):
    """Electric force plot as a special case of inv square force."""

    def __init__(self, Q, q, *args, **kwargs):
        self.Q, self.q = Q, q
        K = 22468879468420441./2500000. # units: Kg m3 / s4 A2
        InvSquForcePlot.__init__(self, *args, v1=Q, v2=q, constant=K, **kwargs)

    def set_title(self):
        self.gca().set_title("Electric, Q={}, q={}".format(self.Q, self.q))


fig, ax = plt.subplots(FigureClass=GravityPlot, M=100, m=10)
fig.visualise()
plt.show()

fig, ax = plt.subplots(FigureClass=ElectricPlot, Q=5, q=2)
fig.visualise()
plt.show()
