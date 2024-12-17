import numpy as np
import triangle

# Define the vertices of the polygon
vertices = np.array([
    [0, 0],
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 0],
    [2, -1]

    , [-5, -3]  # BIG triangle
    , [9, -3]   # BIG triangle
    , [2, 9]    # BIG triangle
])

# Define the segments (edges) that must be included in the triangulation
segments = np.array([
    [0, 1],  # Edge between vertex 0 and vertex 1
    [1, 2],  # Edge between vertex 1 and vertex 2
    [2, 3],  # Edge between vertex 2 and vertex 3
    [3, 4],  # Edge between vertex 3 and vertex 4
    [4, 5],  # Edge between vertex 4 and vertex 5
    [5, 0]  # Edge between vertex 5 and vertex 0
    , [0, 4]      # diagonal

    , [6,7]     # BIG triangel
    , [7,8]     # BIG triangel
    , [8,6]     # BIG triangel
])

# structures for planar graph
class Vertex:
    def __init__(self, idx):
        self.idx = idx
        self.point = None
        self.neighbour_triangles_list = None

    def __hash__(self):
        return hash(self.idx)

graph = {
    Vertex(0) : {Vertex(1), Vertex(4), Vertex(5)},
    Vertex(1) : {Vertex(0), Vertex(2)},
    # ...
}

# Create a dictionary for triangle input
tri_input = {
    'vertices': vertices,
    'segments': segments,
}

# Perform constrained Delaunay triangulation
triangulated_data = triangle.triangulate(tri_input, 'p')

# Extract triangles and plot them
import matplotlib.pyplot as plt

plt.triplot(vertices[:, 0], vertices[:, 1], triangulated_data['triangles'], color='gray', alpha=0.5)
# plt.scatter(vertices[:, 0], vertices[:, 1])
plt.show()
