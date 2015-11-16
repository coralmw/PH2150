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
            # we use partial function application as lambda's are wrong
            # in this application, as they are late binding.
            # http://docs.python-guide.org/en/latest/writing/gotchas/
            button = tk.Button(self, text=button_name,
                               command=partial(self.updatefunc, discription))
            button.grid(column=0, sticky=tk.W+tk.E)
            self.buttons.append(button)


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
