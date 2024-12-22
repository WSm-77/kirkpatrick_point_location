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

def create_planar_subdivision():
    """
    Create a planar subdivision interactively using matplotlib.
    Returns:
    1. A list of all points (bounding polygon and inner points).
    2. A list of tuples of indices representing edges and diagonals between points.
    """
    fig, ax = plt.subplots()
    ax.set_title("Create Planar Subdivision\n1. Add bounding polygon\n2. Right-click to finish polygon\n"
                 "3. Add points inside\n4. Right-click to finish points\n"
                 "5. Add diagonals\n6. Right-click to finish")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    all_points = []  # All entered points (bounding polygon and inner points)
    diagonals = []  # Tuples of indices representing diagonals
    bounding_polygon_end = [False]  # Tracks if the bounding polygon is finished
    diagonal_start_index = [None]  # Tracks the starting point index for a diagonal
    current_stage = [1]  # Tracks the current stage: 1=polygon, 2=points, 3=diagonals

    def onclick(event):
        if event.button == 1:  # Left-click to add a point
            if current_stage[0] == 1:
                # Add to bounding polygon
                all_points.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, 'ro')  # Red points for polygon
                if len(all_points) > 1:
                    ax.plot([all_points[-2][0], all_points[-1][0]],
                            [all_points[-2][1], all_points[-1][1]], 'r-')
                plt.draw()
            elif current_stage[0] == 2:
                # Add inner points
                all_points.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, 'bo')  # Blue points for inner points
                plt.draw()
            elif current_stage[0] == 3:
                # Add diagonals
                for i, point in enumerate(all_points):
                    if abs(event.xdata - point[0]) < 0.5 and abs(event.ydata - point[1]) < 0.5:
                        if diagonal_start_index[0] is None:
                            # Set starting point for diagonal
                            diagonal_start_index[0] = i
                            ax.plot(point[0], point[1], 'go')  # Mark as selected
                            plt.draw()
                        else:
                            # Set ending point for diagonal
                            start = diagonal_start_index[0]
                            end = i
                            if start != end and (start, end) not in diagonals and (end, start) not in diagonals:
                                diagonals.append((start, end))
                                ax.plot([all_points[start][0], all_points[end][0]],
                                        [all_points[start][1], all_points[end][1]], 'g--')
                                plt.draw()
                            diagonal_start_index[0] = None
                        break

        elif event.button == 3:  # Right-click to finish the current stage
            if current_stage[0] == 1:
                # Close the polygon
                if len(all_points) > 2:
                    # all_points.append(all_points[0])  # Close the loop
                    ax.plot([all_points[-1][0], all_points[0][0]],
                            [all_points[-1][1], all_points[0][1]], 'r-')
                    plt.draw()
                    bounding_polygon_end[0] = True
                current_stage[0] = 2
                ax.set_title("Add points inside the polygon\nRight-click to finish points")
                for i in range(len(all_points) - 1):
                    diagonals.append((i, i + 1))
                diagonals.append((0, len(all_points) - 1))
            elif current_stage[0] == 2:
                current_stage[0] = 3
                ax.set_title("Add diagonals\nRight-click to finish")
            elif current_stage[0] == 3:
                plt.close()

    # Connect the event to the figure
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Return the subdivision components
    return all_points, diagonals
