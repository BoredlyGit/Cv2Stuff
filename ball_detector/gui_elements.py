import tkinter as tk
import math


def clone_widget(widget, master, **widget_kwargs):
    # Can't change widget master, so just clone with a new master instead.
    clone_kwargs = {}
    for key in widget.configure():
        clone_kwargs[key] = widget.cget(key)
    clone_kwargs.update(widget_kwargs)
    if "from" in clone_kwargs:  # special case (from is a reserved keyword so tk uses from_).
        clone_kwargs["from_"] = clone_kwargs["from"]
        del clone_kwargs["from"]
    cloned = widget.__class__(master=master, **clone_kwargs)
    # print(f"cls: {widget.__class__},\n kwargs: {clone_kwargs}")
    widget.destroy()
    return cloned


class InputField(tk.Frame):
    def __init__(self, widget, text, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.widget = clone_widget(widget, self)
        self.label = tk.Label(self, text=text)

        self.label.pack(side="left")
        self.widget.pack(side="left")

    @property
    def value(self):
        # This is an abomination
        return self.widget.get() if not ("variable" in self.widget.configure().keys() and self.widget.cget("variable") != "") else self.widget.getvar(self.widget.cget("variable"))


class MultiInputField(tk.Frame):  # not subclassed b/c everything is overridden anyway
    def __init__(self, widgets, text, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(widgets) > 0
        self.widgets = [clone_widget(widget, self) for widget in widgets]

        self.label = tk.Label(self, text=text)

        self.label.pack(side="left")
        for widget in self.widgets:
            widget.pack(side="left")

    @property
    def value(self):
        return [widget.get() for widget in self.widgets]


class OptionsFrame(tk.Frame):
    def __init__(self, init_options=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if init_options is None:
            self.prev_options = {'ball_min_area': 500, 'ball_min_coverage': 0.60, 'img_resize': None, 'subtract_canny': 0, 'hsv_ranges': [{'min': [0, 0, 0], 'max': [255, 255, 255]}], 'name': 'GUI'}
        else:
            self.prev_options = init_options
        self.option_names = ["ball_min_area", "ball_min_coverage", "img_resize", "subtract_canny", "use_metric"]
        self.hsv_mask_selectors = []

        self.ball_area_options = tk.Frame(master=self, relief="solid", borderwidth=1)

        self.ball_min_area = InputField(tk.Spinbox(from_=0, to=math.inf, value=self.prev_options["ball_min_area"]), "ball_min_area: ", master=self.ball_area_options)
        self.ball_min_area.pack()
        self.ball_min_coverage = InputField(tk.Spinbox(from_=0, to=1, increment=0.05, value=self.prev_options["ball_min_coverage"]), "ball_min_coverage: ", master=self.ball_area_options)
        self.ball_min_coverage.pack()

        self.other_options = tk.Frame(master=self,  relief="solid", borderwidth=1)

        self.img_resize = MultiInputField([tk.Spinbox(from_=0, to=math.inf), tk.Spinbox(from_=0, to=math.inf)],
                                          "image_resize ((0, 0) for no resize): ",
                                          master=self.other_options)
        self.img_resize.pack()
        self.subtract_canny = InputField(tk.Checkbutton(variable=tk.BooleanVar()), "subtract_canny: ", master=self.other_options)
        self.subtract_canny.pack()

        self.use_metric = InputField(tk.Checkbutton(variable=tk.BooleanVar()), "use metric: ", master=self.other_options)
        self.use_metric.pack()

        self.HSV_frame = tk.Frame(relief="solid", borderwidth=1)
        self.add_range_button = tk.Button(self.HSV_frame, text="Add HSV range", command=self.add_hsv_range)
        self.add_range_button.pack()
        for hsv_range in self.prev_options["hsv_ranges"]:
            self.add_hsv_range(hsv_range)

        self.ball_area_options.pack(side="left")
        self.other_options.pack(side="left")
        self.HSV_frame.pack(pady=10)

    def add_hsv_range(self, init_range=None):
        selector = HSVMaskSelector(init_range=init_range, master=self.HSV_frame)
        self.hsv_mask_selectors.append(selector)
        selector.pack(side="left")

    @property
    def options(self):
        ret = {op: getattr(self, op).value for op in self.option_names}
        ret["hsv_ranges"] = [mask_sel.value for mask_sel in self.hsv_mask_selectors]
        ret["name"] = "GUI"
        try:
            ret["ball_min_area"] = float(ret["ball_min_area"])
            ret["ball_min_coverage"] = float(ret["ball_min_coverage"])
            ret["img_resize"] = list(int(dim) for dim in ret["img_resize"])
            ret["subtract_canny"] = int(ret["subtract_canny"])
            ret["use_metric"] = int(ret["use_metric"])
        except ValueError:
            print("valerr")
            return self.prev_options

        if 0 in ret["img_resize"]:
            ret["img_resize"] = None
        self.prev_options = ret
        return ret


class HSVMaskSelector(tk.Frame):
    def __init__(self, init_range=None, *args, **kwargs):
        super().__init__(relief="solid", borderwidth=1, *args, **kwargs)
        if init_range is None:
            init_range = {'min': [0, 0, 0], 'max': [255, 255, 255]}

        self.title = tk.Entry(master=self)
        self.title.insert(0, init_range.get("name", ""))
        self.title.pack()

        self.h_min = InputField(tk.Scale(orient="horizontal", from_=0, to=180, sliderlength=15), "H min", self)
        self.s_min = InputField(tk.Scale(orient="horizontal", from_=0, to=255, sliderlength=15), "S min", self)
        self.v_min = InputField(tk.Scale(orient="horizontal", from_=0, to=255, sliderlength=15), "V min", self)
        self.h_max = InputField(tk.Scale(orient="horizontal", from_=0, to=180, sliderlength=15), "H max", self)
        self.s_max = InputField(tk.Scale(orient="horizontal", from_=0, to=255, sliderlength=15), "S max", self)
        self.v_max = InputField(tk.Scale(orient="horizontal", from_=0, to=255, sliderlength=15), "V max", self)

        self.h_min.widget.set(init_range["min"][0])
        self.s_min.widget.set(init_range["min"][1])
        self.v_min.widget.set(init_range["min"][2])
        self.h_max.widget.set(init_range["max"][0])
        self.s_max.widget.set(init_range["max"][1])
        self.v_max.widget.set(init_range["max"][2])

        self.h_min.pack()
        self.s_min.pack()
        self.v_min.pack()
        self.h_max.pack()
        self.s_max.pack()
        self.v_max.pack()

    @property
    def value(self):
        ret = {}
        for bound in ("min", "max"):
            ret[bound] = [getattr(self, f"{val}_{bound}").value for val in ("h", "s", "v")]
        return ret


# of = OptionsFrame()
# of.pack()
# while True:
#     of.update()
#     print(of.options)
