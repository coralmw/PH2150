import Tkinter as tk
import tkMessageBox

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
        '''gets a axis out of {x,y,z}.
        returns a number.
        '''
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
        '''sets the axis {x,y,z} to the given value.'''
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
    '''Class represting the plot portion of the main window.
    Handels displaying the plot, and updating when required.
    '''

    def __init__(self, master=None):
        self.master = master
        tk.Frame.__init__(self, self.master)

        # create the figure to plot into
        self.fig = Figure(figsize=(5,4), dpi=100) # RETINA BRO?
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.show() # and display a placeholder box

        self.canvas.get_tk_widget().grid(row = 0, column = 1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.toolbar.grid(row = 7, column = 1)

        self.plotLock = threading.Lock() # lock that protects the plot from mixups

        self.ax = mplot3d.Axes3D(self.fig)
        self.ax.set_title("Path of charged particle under influence of electric and magnetic fields")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z') # set this here to avoid excess time spent in the
                                # update method


    def PlotElectron(self, params):
        '''This function calculates the path of the electron and displays it.
        basically renterant, for multithreading.
        params is a dict, with format deatiled in the input class
        '''
        #print params

        # extract needed values from the params dict
        E, B = params['E'].v, params['B'].v # E, B are Vector objects, we just need
                                            # the numpy arrays
        Particle = params['particle']
        ParticleE, ParticleRK = Particle.copy(), Particle.copy() # sep particles for each
                                                                 # DE method
        T, dt = params['MaxT'], params['dt']

        # print E, B, ParticleE, T, dt

        if params['Euler']:
            Peuler = Euler(E, B, ParticleE, Tmax=T, dt=dt)
        if params['RK']:
            Prk = RK(E, B, ParticleRK, Tmax=T, dt=dt)

        # print 'plotted'
        # print Peuler[-1]
        self.plotLock.acquire()
        self.ax.cla() # clear the plot
        if params['Euler']:
            self.ax.plot3D(color='blue', label='Euler', **VectorListToPLOT3D(Peuler))
        if params['RK']:
            self.ax.plot3D(color='red', label='RK', **VectorListToPLOT3D(Prk))
        self.canvas.show()
        self.plotLock.release()

    def update(self, params):
        '''Calls PlotElectron in a new thread, so as to not block the UI.'''
        threading.Thread(target=self.PlotElectron, args=(params,)).start()
        #self.PlotElectron(params)


class InputRow(tk.Frame):
    '''Class representing each row of the input panel.
    '''

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
        '''Gets the underlying scale bar in the frame.'''
        return self.bar

class DiscrMethodsButtons(tk.Frame):

    def __init__(self, master):
        self.master = master
        tk.Frame.__init__(self, self.master)

        eulerTxt = """The euler method is a first order NDE solver.
It\'s error goes with the step size, rather than mush faster in the case of RK.
We find the tangent line, and step the sol'n down this line."""

        RKtxt = """This method uses a midpoint to cancel out erros from first order
solves. Devloped in the 1900's by C. Runge and M. W. Kutta. The error goes with the
step size to the 4th power."""

        self.EluerB = tk.Button(
                                self,
                                text='Euler',
                                command=partial(
                                                tkMessageBox.showinfo,
                                                'Euler method',
                                                eulerTxt,
                                               ),
                                )
        self.EluerB.grid(row=0, column=0)

        self.RKB = tk.Button(
                              self,
                              text='Rk',
                              command=partial(
                                              tkMessageBox.showinfo,
                                              'RK method',
                                              RKtxt,
                                             ),
                             )
        self.RKB.grid(row=0, column=1)





class InputFrame(tk.Frame):
    '''Class representing the controls portion of the main window.
    Contains the main application logic, as all the logic pertains
    to the buttons being manipulated.
    '''

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

        self.methods = DiscrMethodsButtons(self)
        self.methods.grid(row=7, column=1)


    def updateWrap(self):
        '''Configures does the final dict updates and passes it to the update function.
        '''
        self.ParamDict['RK'] = self.RKEnabled.get()
        self.ParamDict['Euler'] = self.EulerEnabled.get()
        self.updateFunc(self.ParamDict)

    def quickUpdateWrap(self):
        '''does the final dict updates and passes it to the update function,
        but does it for a much larger timestep suitible for live updates.
        '''
        self.ParamDict['RK'] = self.RKEnabled.get()
        self.ParamDict['Euler'] = self.EulerEnabled.get()
        dtOld = self.ParamDict['dt']
        self.ParamDict['dt'] = self.ParamDict['dt'] * 50.
        self.updateFunc(self.ParamDict)
        self.ParamDict['dt'] = dtOld # restore the old dt after done

    def ScalebarUpdate(self, nameDirPair, value):
        '''Called when a scalebar is modified. Called with the name of the
        bar by the partial func application above. Stores the new value in the
        config dict.
        '''
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
        self.quickUpdateWrap() # do a fastish update


class App(tk.Frame):
    '''Root class that makes the seperate frames. the input window and the plot
    communicate only via the update function.
    '''

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
