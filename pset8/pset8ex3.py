import Tkinter as tk
from functools import partial

class DiscrButton(object):

    def __init__(self, discr, pos, func=None):
        self.text = discr
        self.pos = pos
        self.func = func

class ButtonFrame(tk.Frame):

    def __init__(self, master=None, textboxUpdate=None):
        self.master = master
        tk.Frame.__init__(self, self.master)

        self.grid()
        self.updatefunc = textboxUpdate

        self.discriptions = {
                            'button':("Buttons can contain text or images,"
                                      " and you can associate a Python function or method with each button."
                                      " When the button is pressed,"
                                      " Tkinter automatically calls that function or method.\n\n"
                                      "The button can only display text in a single font,"
                                      " but the text may span more than one line."
                                      " In addition, one of the characters can be underlined,"
                                      " for example to mark a keyboard shortcut."
                                      " By default, the Tab key can be used to move to a button widget."),
                            'Canvas':'Another discr',
                            'Entry':'11111',
                            'message':'55555',
                            'Text':'Text Boxes are awesom',
                            'Pack Manager':'Is a pain in the butt'
                            }

        self.make_discr_buttons()


    def make_discr_buttons(self):
        for button_name, discription in self.discriptions.items():
            print button_name, discription
            # we use partial function application as lambda's are wrong
            # in this application, as they are late binding.
            # http://docs.python-guide.org/en/latest/writing/gotchas/
            button = tk.Button(self, text=button_name,
                               command=partial(self.updatefunc, discription))
            button.grid(column=0)





class App(tk.Frame):

    StickyAll = tk.W+tk.E+tk.N+tk.S

    def __init__(self, master=None):
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

        self.ButtonFrame = ButtonFrame(self, textboxUpdate=self._change_discr_text)
        self.ButtonFrame.grid(row=0, column=0)

        self.QuitB = tk.Button(self, text='Quit', command=self.quit)
        self.QuitB.grid(row=1, column=0)

    def _change_discr_text(self, text):
        '''updates the discr text box. you need to make it editable before inserting
        '''
        self.DiscrText.config(state=tk.NORMAL)
        self.DiscrText.delete('1.0', tk.END) # everything delete.
        self.DiscrText.insert(tk.END, text)  # WHY IS THE ORDER OF ARGS OPPO?
        self.DiscrText.config(state=tk.DISABLED)


root = tk.Tk()
# root.geometry("700x200+200+200")
app = App(master=root)
app.master.title('Sample application')
app.mainloop()
