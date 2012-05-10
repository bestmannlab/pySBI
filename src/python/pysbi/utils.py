from matplotlib.backends.backend_agg import FigureCanvasAgg

class Struct():
    def __init__(self):
        pass

def save_to_png(fig, output_file):
    fig.set_facecolor("#FFFFFF")
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(output_file, dpi=72)
