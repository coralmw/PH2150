import numpy as np
import matplotlib.pyplot as plt

class VelocityProfile(object):

    def __init__(self, R, beta, mu0, n):
        self.R = float(R)
        self.beta = float(beta)
        self.mu0 = float(mu0)
        self.n = float(n)

    def _v_loc(self, r):
        '''calculates the velocity at a distance r from the center of the pipe.
        private function.
        '''
        if type(r) == np.ndarray and (r > self.R).any():
            raise ValueError('r must be within the pipe.')

        prefactor = (self.beta/(2*self.mu0))**(1./self.n)
        middle = self.n/(self.n+1)
        rdep = ( self.R**(1+1/self.n) - abs(r)**(1+1/self.n) )

        return prefactor*middle*rdep

    def vProfile(self, res=1000):
        rarr = np.linspace(-self.R, self.R, res)
        varr = self._v_loc(rarr)
        return rarr, varr


smallPipe = VelocityProfile(1, 0.06, 0.02, 0.1)
largePipe = VelocityProfile(2, 0.03, 0.02, 0.1)
plt.plot(*smallPipe.vProfile(), label='small pipe')
plt.plot(*largePipe.vProfile(), label='large pipe')
plt.legend()
plt.xlabel('r')
plt.ylabel('velocity')

plt.show()


#### EX2

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

######## Pset8ex3

import Tkinter as tk
import tkMessageBox
from functools import partial
import simplejson as json
import os

class AppStrings(object):
    '''Class represting all the user strings used in the app,
    to allow for localisation.
    '''

    def __init__(self, loc):

        pypath = os.path.dirname(os.path.abspath(__file__))
        strPath = os.path.join(pypath, 'discr_{}.json'.format(loc))
        with open(strPath, 'rb') as f:
            self.strings = json.loads(f.read())

    def __getattr__(self, attr):
         return self.strings[attr]


# class DiscrButton(object):
#
#     def __init__(self, discr, pos, func=None):
#         self.text = discr
#         self.pos = pos
#         self.func = func

class ButtonFrame(tk.Frame):

    def __init__(self, master=None, textboxUpdate=None, strings=None):
        self.strings = strings
        self.master = master
        tk.Frame.__init__(self, self.master)

        self.grid()
        self.updatefunc = textboxUpdate

        self.buttonNames = [
                            'button',
                            'Canvas',
                            'Entry',
                            'message',
                            'Text',
                            'Pack Manager',
                           ]

        self.buttons = []
        self.make_discr_buttons()

    def make_discr_buttons(self):
        for button_name in self.buttonNames:
            discription = self.strings.__getattr__(button_name)
            print button_name, discription
            # we use partial function application as lambda's are wrong
            # in this application, as they are late binding.
            # http://docs.python-guide.org/en/latest/writing/gotchas/
            button = tk.Button(self, text=button_name,
                               command=partial(self.updatefunc, discription))
            button.grid(column=0)
            self.buttons.append(button)

        # Maxwidth = max([b.config()['width'] for b in self.buttons])
        # print [b.config() for b in self.buttons]
        #
        # for button in self.buttons:
        #     button.config(width=Maxwidth)


class AppButtonFrame(tk.Frame):

    def __init__(self, master=None, textboxUpdate=None, strings=None):
        self.strings = strings
        self.master = master
        tk.Frame.__init__(self, self.master)

        self.grid()

        self.QuitB = tk.Button(self, text=strings.quit, command=self.quit)
        self.QuitB.grid(row=0, column=0)

        # create a button to show strings.licence info
        self.licenceB = tk.Button(
                                  self,
                                  text=strings.licence,
                                  command=partial(
                                                  tkMessageBox.showinfo,
                                                  strings.licence,
                                                  strings.licenceText
                                                 ),
                                 )
        self.licenceB.grid(row=0, column=1)


class App(tk.Frame):

    StickyAll = tk.W+tk.E+tk.N+tk.S

    def __init__(self, master=None, strings=None):
        self.strings = strings
        self.master = master
        tk.Frame.__init__(self, self.master)
        self.grid()
        self.master.title("Grid Manager")

        #
        # self.TextFrame = tk.Frame(master, bg="green")
        # self.TextFrame.grid(row=0, column=2, rowspan=6, columnspan=3, sticky=self.StickyAll)

        self.DiscrText = tk.Text(self, height=10, wrap=tk.WORD, width=60)
        self.DiscrText.grid(row=0, column=1, sticky=self.StickyAll)
        self.DiscrText.insert(tk.END, "Just a text Widget\nin two lines\n")
        self.DiscrText.config(state=tk.DISABLED) # must be returned to normal for
                                                 # insert calls to work

        # make the buttons, pass _change_discr_text as a callback
        self.ButtonFrame = ButtonFrame(self,
                                       textboxUpdate=self._change_discr_text,
                                       strings=self.strings
                                      )
        self.ButtonFrame.grid(row=1, column=0)

        self.AppButtonFrame = AppButtonFrame(self, strings=self.strings)
        self.ButtonFrame.grid(row=0, column=0)

    def _change_discr_text(self, text):
        '''updates the discr text box. you need to make it editable before inserting
        '''
        self.DiscrText.config(state=tk.NORMAL)
        self.DiscrText.delete('1.0', tk.END) # everything delete.
        self.DiscrText.insert(tk.END, text)  # WHY IS THE ORDER OF ARGS OPPO?
        self.DiscrText.config(state=tk.DISABLED)


def main():
    strings = AppStrings(loc='en')
    root = tk.Tk()
    # root.geometry("700x200+200+200")
    app = App(master=root, strings=strings)
    app.master.title('Sample application')
    app.mainloop()


if __name__ == '__main__':
    main()


######### pset8ex3 strings.json file

{
 "Canvas": "Is what you put things on",
 "Text": "Text Boxes are awesom, as demonstrated.",
 "button": "Buttons can contain text or images, and you can associate a Python function or method with each button. When the button is pressed, Tkinter automatically calls that function or method.\n\nThe button can only display text in a single font, but the text may span more than one line. In addition, one of the characters can be underlined, for example to mark a keyboard shortcut. By default, the Tab key can be used to move to a button widget.",
 "Entry": "An entry box is a single line text box that allows the user to enter text. \nIt supports passwords too.",
 "message": "Message dialogs",
 "Pack Manager": "Shoves things into the wallls",
 "quit": "Quit",
 "licence":"Licence",
 "licenceText":"Licence: GPL (I wish), RHUL probably owns this.\nCC Thomas Parks 2015"
}


######## pset8ex4.py

import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
from functools import partial
import numpy as np
import threading

from DEsolvers import Particle, Euler, RK, VectorListToPLOT3D


class Vector(object):
    '''A class for vectors representing fields.
    '''

    def __init__(self, x=0, y=0, z=0):
        # needs to be a float or everything is blank
        self.v = np.array([x, y, z], dtype=np.float32)

    def getAxis(self, dir):
        if dir == 'x':
            return self.v[0]
        elif dir == 'y':
            return self.v[1]
        elif dir == 'z':
            return self.v[2]
        else:
            raise AttributeError('{} not found'.format(key))

    def __repr__(self):
        return self.v.__repr__()

    def setAxis(self, dir, val):
        print 'setting', dir, 'to', val
        if dir == 'x':
            self.v[0] = val
        elif dir == 'y':
            self.v[1] = val
        elif dir == 'z':
            self.v[2] = val
        else:
            raise AttributeError('{} not found'.format(key))


class PlotFrame(tk.Frame):

    def __init__(self, master=None):
        self.master = master
        tk.Frame.__init__(self, self.master)

        self.fig = Figure(figsize=(5,4), dpi=100) # RETINA BRO?
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.show()

        self.canvas.get_tk_widget().grid(row = 0, column = 1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.toolbar.grid(row = 7, column = 1)

        self.plotLock = threading.Lock()

        self.ax = mplot3d.Axes3D(self.fig)
        self.ax.set_title("Path of charged particle under influence of electric and magnetic fields")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')


    def PlotElectronShit(self, params):
        print params

        E, B = params['E'].v, params['B'].v
        Particle = params['particle']
        ParticleE, ParticleRK = Particle.copy(), Particle.copy()
        T, dt = params['MaxT'], params['dt']

        # print E, B, ParticleE, T, dt

        if params['Euler']:
            Peuler = Euler(E, B, ParticleE, Tmax=T, dt=dt)
        if params['RK']:
            Prk = RK(E, B, ParticleRK, Tmax=T, dt=dt)

        # print 'plotted'
        # print Peuler[-1]
        self.plotLock.acquire()
        self.ax.cla()
        if params['Euler']:
            self.ax.plot3D(color='blue', label='Euler', **VectorListToPLOT3D(Peuler))
        if params['RK']:
            self.ax.plot3D(color='red', label='RK', **VectorListToPLOT3D(Prk))
        self.canvas.show()
        self.plotLock.release()

    def update(self, params):
        threading.Thread(target=self.PlotElectronShit, args=(params,)).start()
        #self.PlotElectronShit(params)


class InputRow(tk.Frame):

        def __init__(self, master, name, **config):
            self.master = master
            self.name = name
            tk.Frame.__init__(self, self.master)
            self.label = tk.Label(self, text=name)
            self.label.grid(row=0, column=0)
            self.bar = tk.Scale(self, from_=-10, to=10,
                                orient=tk.HORIZONTAL,
                                resolution=0.01
                               )
            self.bar.config(**config)
            self.bar.grid(row=0, column=1)

        def GetScaleBar(self):
            return self.bar


class InputFrame(tk.Frame):

    def __init__(self, master, updateFunc=None):
        self.master = master
        self.updateFunc = updateFunc
        tk.Frame.__init__(self, self.master)

        self.QuitB = tk.Button(self, text='quit', command=self.quit)
        self.QuitB.grid(row=0, column=0)

        self.RKEnabled = tk.IntVar()
        self.EulerEnabled = tk.IntVar()


        self.ParamDict = {} # this dict will be passed to the plotting func

        initParams = {
                  'E':Vector(0., 2., 0.),
                  'B':Vector(0., 0., 4.),
                  'V':Vector(20., 10., 2.),
                  'P':Vector(0., 0., 0.),
                  'MaxT':6,
                  'dt':0.01,
                  'q':5.0,
                  'm':5.0,
                 }

        self.AdjustParams = {
                     'E':Vector(),
                     'B':Vector(),
                     'P':Vector(),
                     'V':Vector(),
                    }

        for name, VectorDict in self.AdjustParams.iteritems():
            for dir in ['x', 'y', 'z']:
                inputRow = InputRow(self, '{}.{}'.format(name, dir))
                inputRow.GetScaleBar().config(command=partial(self.ScalebarUpdate,
                                                              (name, dir))
                                             )
                if name in initParams:
                    inputRow.GetScaleBar().set(initParams[name].getAxis(dir))
                inputRow.grid(column=0)

        self.SclarParams = {'m':0, 'q':5, 'dt':0.001, 'MaxT':5}
        for i, (name, VectorDict) in enumerate(self.SclarParams.iteritems()):
            inputRow = InputRow(self, '{}'.format(name))
            inputRow.GetScaleBar().config(command=partial(self.ScalebarUpdate,
                                                          (name, None))
                                         )
            if name == 'dt':
                inputRow.GetScaleBar().config(from_=0.0001, to=0.1, resolution=0.0001)
            if name in initParams:
                inputRow.GetScaleBar().set(initParams[name])
            inputRow.grid(row=i+1, column=1)

        self.UpdateB = tk.Button(self, text='update',
                                 command=self.updateWrap)
        self.UpdateB.grid(row=12, column=1)

        self.RKbox = tk.Checkbutton(self, text="RK", variable=self.RKEnabled)
        self.Eulerbox = tk.Checkbutton(self, text="Euler", variable=self.EulerEnabled)
        self.RKbox.grid(row=5, column=1)
        self.Eulerbox.grid(row=6, column=1)

    def updateWrap(self):
        self.ParamDict['RK'] = self.RKEnabled.get()
        self.ParamDict['Euler'] = self.EulerEnabled.get()
        self.updateFunc(self.ParamDict)

    def quickUpdateWrap(self):
        self.ParamDict['RK'] = self.RKEnabled.get()
        self.ParamDict['Euler'] = self.EulerEnabled.get()
        dtOld = self.ParamDict['dt']
        self.ParamDict['dt'] = self.ParamDict['dt'] * 50.
        self.updateFunc(self.ParamDict)
        self.ParamDict['dt'] = dtOld

    def ScalebarUpdate(self, nameDirPair, value):
        name, direction = nameDirPair
        if direction:
            self.AdjustParams[name].setAxis(direction, float(value))
        else:
            self.SclarParams[name] = float(value)

        ParamDict = self.AdjustParams.copy()
        ParamDict.update(self.SclarParams)
        ParamDict['particle'] = Particle(V=self.AdjustParams['V'].v,
                                         P=self.AdjustParams['P'].v,
                                         m=self.SclarParams['m'],
                                         q=self.SclarParams['q']
                                        )

        self.ParamDict.update(ParamDict)
        self.quickUpdateWrap()


class App(tk.Frame):

    def __init__(self, master):
        self.master = master
        tk.Frame.__init__(self, self.master)

        # make the plot frame
        self.plotFrame = PlotFrame(master=None)
        self.plotFrame.grid(row=0, column=1)

        # make and position the controls frame
        # keep the isolation bwteen classes by only providing the update func
        self.inputFrame = InputFrame(master=None, updateFunc=self.plotFrame.update)
        self.inputFrame.grid(row=0, column=0)


def main():
    root = tk.Tk()
    # root.geometry("700x200+200+200")
    app = App(master=root)
    app.master.title('Numerical Solutions to the electron\'s path')
    app.mainloop()


if __name__ == '__main__':
    main()


######## pset8 ex3 DEsoolve.py

'''Module for doing RK solving of a electrons motion in electric
and magnetic spaces.

CC Thomas Parks 2015
thomasparks@outlook.com
thomas.parks.2013@live.rhul.ac.uk

Lisenced under MIT availabe from https://opensource.org/licenses/MIT
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def VectorListToPLOT3D(vecs):
    '''Returns a form suitable for plot3d.'''
    return dict(zip(
                     ['xs','ys','zs'],
                     [dim.flatten() for dim in np.hsplit(np.array(vecs), 3)]
                    ))


class Particle(object):

    def __init__(self, V, P, q, m):
        if type(V) != np.ndarray:
            print V, type(V)
            raise ValueError('V is the wrong kind or shape')

        if type(P) != np.ndarray:
            print P, type(P)
            raise ValueError('P is the wrong kind or shape')

        if type(q) not in (float, int):
            print type(q)
            print q
            print 'HERRRRRRE'

        assert(isinstance(q, (int, long, float, complex)))
        assert(isinstance(m, (int, long, float, complex)))

        self.V = V
        self.P = P
        self.m = m
        self.q = q

    def copy(self):
        return Particle(self.V.copy(), self.P.copy(), self.m, self.q)

    def __repr__(self):
        return "Position:{}, V:{}".format(self.P, self.V)


def ForceAtVelocity(E, B, particle):
    '''find the force at a given vector location.
    takes vectors, E,B and a scalr q.
    returns vector of the force.'''
    return particle.q * (E + np.cross(particle.V, B)) # Faraday's law


def Euler(E, B, particle, Tmax=10, dt=0.001):
    assert(type(E) == np.ndarray)
    assert(type(B) == np.ndarray)
    Peuler = []
    t = 0.

    while t < Tmax: # trace path until time reaches value e.g. 10
        f = ForceAtVelocity(E, B, particle)
        a = f / particle.m
        dv = a * dt
        dp = (particle.V+dv)*dt # do a iteration using eulers method

        particle.V += dv
        particle.P += dp
        Peuler.append(particle.P.copy())
        t += dt

    return Peuler


def RK(E, B, particle, Tmax=10, dt=0.001):
    Prk = []
    t = 0.

    while t < Tmax: # trace path until time reaches value e.g. 10
        # first find the first-order update
        f1 = ForceAtVelocity(E, B, particle)
        a1 = f1 / particle.m
        dv1 = a1 * dt
        dp1 = (particle.V+dv1) * dt # just using this gives the Euler method

        # find the force at the first midpoint.
        midParticle1 = particle.copy()
        midParticle1.V += (dv1/2)
        f2 = ForceAtVelocity(E, B, midParticle1)
        a2 = f2 / particle.m
        dv2 = a2 * dt
        dp2 = (particle.V+dv2) * dt

        # second midpoint and so on
        midParticle2 = particle.copy()
        midParticle2.V += (dv2/2)
        f3 = ForceAtVelocity(E, B, midParticle1)
        a3 = f3 / particle.m
        dv3 = a3 * dt
        dp3 = (particle.V+dv3) * dt

        midParticle3 = particle.copy()
        midParticle3.V += dv3
        f4 = ForceAtVelocity(E, B, midParticle3)
        a4 = f4 / particle.m
        dv4 = a4 * dt
        dp4 = (particle.V+dv4) * dt

        particle.V += dv1/6. + dv2/3. + dv3/3. + dv4/6. # use RK for both V and P, for MOR ACCUR.
        particle.P += dp1/6. + dp2/3. + dp3/3. + dp4/6.
        Prk.append(particle.P.copy())
        t += dt

    return Prk




if __name__ == '__main__':

    E = np.array((0., 2., 0.)) # all vectors for neatness
    B = np.array((0., 0., 4.))
    P = np.array((0., 0., 0.))
    V = np.array((20., 10., 2.))

    m = 2.0 # Mass of the particle
    q = 5.0 # Charge
    t = 0.0
    T = 6
    dt = 0.001

    ParticleE = Particle(V, P, m, q)
    ParticleRK = ParticleE.copy()

    Peuler, Prk = Euler(E, B, ParticleE, Tmax=T, dt=dt), RK(E, B, ParticleRK, Tmax=T, dt=dt)
    print Peuler

    fig1=plt.figure()
    ax1 = fig1.add_subplot(1,1,1, projection='3d')
    ax1.set_title("Path of charged particle under influence of electric and magnetic fields")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.plot3D(color='blue', label='Euler', **VectorListToPLOT3D(Peuler))
    ax1.plot3D(color='red', label='RK', **VectorListToPLOT3D(Prk))

    plt.show()
