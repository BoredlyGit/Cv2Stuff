import tkinter as tk
import math


class InputField(tk.Frame):
    def __init__(self, widget, text, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Can't change widget master, so just clone it instead.
        clone_args = {}
        for key in widget.configure():
            clone_args[key] = widget.cget(key)
        cloned = widget.__class__(master=self, **clone_args)
        cloned.master = self
        widget.destroy()

        self.widget = cloned
        self.label = tk.Label(self, text=text)

        self.label.pack(side="left")
        self.widget.pack(side="left")
        print(self.widget.__class__, self.widget.configure())

    @property
    def value(self):
        return self.widget.get()


class MultiInputField(tk.Frame):  # not subclassed b/c everything is overridden anyways
    def __init__(self, widgets, text, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.widgets = []
        assert len(widgets) > 0
        for widget in widgets:
            # Can't change widget master, so just clone it instead.
            clone_args = {}
            for key in widget.configure():
                clone_args[key] = widget.cget(key)
            cloned = widget.__class__(master=self, **clone_args)
            cloned.master = self
            self.widgets.append(cloned)
            widget.destroy()

        self.label = tk.Label(self, text=text)

        self.label.pack(side="left")
        for widget in self.widgets:
            widget.pack(side="left")
            print(widget.__class__, widget.configure())

    @property
    def value(self):
        return [widget.get() for widget in self.widgets]


class OptionsFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.option_names = ["ball_min_area", "ball_min_coverage", "image_resize"]

        self.ball_area_options = tk.LabelFrame(master=self)
        self.ball_area_options.pack(side="left")

        self.ball_min_area = InputField(tk.Spinbox(from_=0, to=math.inf), "ball_min_area: ", master=self.ball_area_options)
        self.ball_min_area.pack()
        self.ball_min_coverage = InputField(tk.Spinbox(from_=0, to=1, increment=0.05), "ball_min_coverage: ", master=self.ball_area_options)
        self.ball_min_coverage.pack()

        self.other_options = tk.LabelFrame(master=self)
        self.other_options.pack(side="left", padx=10)

        self.image_resize = MultiInputField([tk.Spinbox(from_=0, to=math.inf),
                                             tk.Spinbox(from_=0, to=math.inf)],
                                            "image_resize ((0, 0) for no resize): ",
                                            master=self.other_options)
        self.image_resize.pack()
        self.subtract_canny = InputField(tk.Checkbutton(), "subtract_canny: ", master=self.other_options)
        self.subtract_canny.pack()

    @property
    def options(self):
        # TODO
        raise NotImplemented

    def validate_options(self):
        pass


class HSVSelector(tk.Frame):
    pass  # TODO


of = OptionsFrame()
of.pack()
while True:
    of.update()
