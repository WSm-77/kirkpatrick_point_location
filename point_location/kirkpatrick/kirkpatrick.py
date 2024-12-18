import triangle

class Kirkpatrick:
    def __init__(self, input_points: list[tuple[float, float]], intput_edges: list[tuple[int, int]]):
        self.input_points = input_points
        self.intput_edges = intput_edges

        self.bounding_triangle = self.get_bounding_triangle(input_points)

        self.all_points = self.input_points + self.bounding_triangle

        all_points_cnt = len(self.all_points)

        # last 3 points are points of bounding triangle
        self.bounding_triangle_points_indices_set = {all_points_cnt - i - 1 for i in range(3)}

        bounding_triangle_edges = [
            (all_points_cnt - 3, all_points_cnt - 2),
            (all_points_cnt - 2, all_points_cnt - 1),
            (all_points_cnt - 1, all_points_cnt - 3)
        ]

        tri_input = {
            'vertices': self.all_points,
            'segments': self.intput_edges + bounding_triangle_edges,
        }

        triangulation_data = triangle.triangulate(tri_input, 'p')
        self.base_triangulation = triangulation_data["triangles"]

    def get_bounding_triangle(self, input_points) -> list[tuple[float, float]]:
        # TODO: implement

        points = [
            (-5, -3),  # BIG triangle
            (9, -3),   # BIG triangle
            (2, 9)     # BIG triangle
        ]

        return points

if __name__ == "__main__":
    # Define the vertices of the polygon
    vertices = [
        (0, 0),
        (1, 2),
        (2, 1),
        (3, 3),
        (4, 0),
        (2, -1)
    ]

    # Define the segments (edges) that must be included in the triangulation
    segments = [
        (0, 1),  # Edge between vertex 0 and vertex 1
        (1, 2),  # Edge between vertex 1 and vertex 2
        (2, 3),  # Edge between vertex 2 and vertex 3
        (3, 4),  # Edge between vertex 3 and vertex 4
        (4, 5),  # Edge between vertex 4 and vertex 5
        (5, 0)  # Edge between vertex 5 and vertex 0
        , (0, 4)      # diagonal
    ]


    # kirkpatrick = Kirkpatrick(list(map(tuple, list(vertices))), list(map(tuple, list(segments))))
    kirkpatrick = Kirkpatrick(vertices, segments)
    print(kirkpatrick.all_points)
    print(kirkpatrick.bounding_triangle_points_indices_set)
    print(kirkpatrick.base_triangulation)

    # Extract triangles and plot them
    import matplotlib.pyplot as plt

    plt.triplot([x[0] for x in kirkpatrick.all_points], [x[1] for x in kirkpatrick.all_points], kirkpatrick.base_triangulation, color='gray', alpha=0.5)
    plt.show()
