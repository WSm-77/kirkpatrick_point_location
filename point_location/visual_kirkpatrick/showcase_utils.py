import matplotlib.pyplot as plt
from point_location.visual_kirkpatrick.visual_kirkpatrick import VisualKirkpatrick

def pick_point(kirkpatrick: VisualKirkpatrick):
    """
    Reads point entered by user with mouse and returns it as list of single point reprezented as tuple of two floats.
    Inside Jupyter Notebook use with ***%matplotlib ipympl***

    :return list of tuple: List containing single point (x, y).
    """
    def draw_planar_subdivision():
        for vertex, neighbour in kirkpatrick.input_edges:
            vertex_object = kirkpatrick.idx_to_point(vertex)
            neighbour_object = kirkpatrick.idx_to_point(neighbour)

            ax.plot([vertex_object.x, neighbour_object.x], [vertex_object.y, neighbour_object.y], color = "blue")

            plt.draw()

    # Event handler for mouse click
    def on_click(event):
        # Check if it's a left-click within the axes
        if event.button == 1 and event.inaxes == ax:
            # Append the point and update the plot
            currX, currY = event.xdata, event.ydata
            ax.scatter(currX, currY, color = "orange", marker = 'o')  # Mark the point

            point.append((currX, currY))

            plt.draw()

            fig.canvas.mpl_disconnect(cid)  # Disconnect the event

    # Connect the event handler

    plt.close('all')    # close all opened plots

    point = []
    fig, ax = plt.subplots(num = "pick point")
    ax.set_title("Left-click to select point to locate")

    draw_planar_subdivision()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    return point
