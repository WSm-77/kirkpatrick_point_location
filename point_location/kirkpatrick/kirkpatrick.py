import networkx as nx
import triangle
from planegeometry.structures.planarmaps import PlanarMap, Point, Segment

class Kirkpatrick:
    def __init__(self, input_points: list[tuple[float, float]], intput_edges: list[tuple[int, int]]):
        self.input_points = input_points
        self.intput_edges = intput_edges

        planar_subdivision = nx.Graph()
        planar_subdivision.add_edges_from(self.intput_edges)

        is_planar, embedding = nx.check_planarity(planar_subdivision)

        if not is_planar:
            raise ValueError("Tis subdivision is not planar!!!")

        self.input_faces = self.get_faces(intput_edges, embedding.get_data())

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

        self.triangulation_data = triangle.triangulate(tri_input, 'p')
        self.base_triangulation = self.triangulation_data['triangles']

    def get_bounding_triangle(self, input_points: list[tuple[float, float]]) -> list[tuple[float, float]]:

        def find_extremes(input_points: list[tuple[float, float]]) -> list[float]:
            inf = float('inf')
            xmin, ymin = inf, inf
            xmax, ymax = -inf, -inf
            for x,y in input_points:
                if x < xmin: xmin = x
                if x > xmax: xmax = x
                if y < ymin: ymin = y
                if y > ymax: ymax = y
            return [xmin, ymin, xmax, ymax]

        xmin, ymin, xmax, ymax = find_extremes(input_points)
        x = xmax - xmin
        y = ymax - ymin

        # Defining points for triangle with no spacing - minimal triangle
        # a = (x*y/2) ** (1/2)
        # points = [
        #     (xmin - a, ymin),  # BIG triangle
        #     (xmax + a, ymin),   # BIG triangle
        #     ((xmin + xmax)/2, ymax + a)     # BIG triangle
        # ]

        # Defining points for triangle with space
        a = (x*y/2 + x + y + 2) ** (1/2)
        points = [
            (xmin - a, ymin - 1),
            (xmax + a, ymin -1),
            ((xmin + xmax)/2, ymax + a)
        ]

        return points

    def get_planar_map(self, triangulation:triangle.triangulate) -> PlanarMap:
        planar_map = PlanarMap()
        points = {i : Point(point[0], point[1]) for i, point in enumerate(triangulation['vertices'])}
        edges = set()

        get_edge = lambda x,y: (min(x,y), max(x,y))

        for triangle in triangulation['triangles']:
            a, b, c = tuple(triangle)
            edges.add(get_edge(a,b))
            edges.add(get_edge(b,c))
            edges.add(get_edge(a,c))

        segments = [Segment(points[i], points[j]) for i,j in edges]
        for segment in segments:
            planar_map.add_edge(segment)

        return planar_map

    def get_faces(self, edges, embedding) -> list[tuple[int, int]]:
        """
        Returns faces of planar subdivision

        :param edges: is an undirected graph as a list of undirected edges
        :param embedding: is a combinatorial embedding dictionary. Format: v1:[v2,v3], v2:[v1], v3:[v1] - clockwise
        ordering of neighbors at each vertex.

        :return: list of faces represented as list of edges
        """

        # Establish set of possible edges
        edgeset = set()
        for edge in edges: # edges is an undirected graph as a set of undirected edges
            edge = list(edge)
            edgeset |= set([(edge[0],edge[1]),(edge[1],edge[0])])

        # Storage for face paths
        faces = []
        path  = []
        for edge in edgeset:
            path.append(edge)
            edgeset -= set([edge])
            break  # (Only one iteration)

        # Trace faces
        while (len(edgeset) > 0):
            neighbors = embedding[path[-1][-1]]
            next_node = neighbors[(neighbors.index(path[-1][-2])+1) % len(neighbors)]
            tup = (path[-1][-1], next_node)
            if tup == path[0]:
                faces.append(path)
                path = []
                for edge in edgeset:
                    path.append(edge)
                    edgeset -= set([edge])
                    break  # (Only one iteration)
            else:
                path.append(tup)
                edgeset -= set([tup])

        if (len(path) != 0):
            faces.append(path)

        return faces

if __name__ == "__main__":
    import json
    import os

    test_dir = "test_input"
    test_name = "test2.json"

    with open(os.path.join(test_dir, test_name), 'r') as file:
        data = json.load(file)

    vertices = data["vertices"]
    segments = data["segments"]

    # kirkpatrick = Kirkpatrick(list(map(tuple, list(vertices))), list(map(tuple, list(segments))))
    kirkpatrick = Kirkpatrick(vertices, segments)
    print(kirkpatrick.all_points)
    print(kirkpatrick.bounding_triangle_points_indices_set)
    print(kirkpatrick.base_triangulation)
    print(*kirkpatrick.input_faces, sep="\n")
    planar_map = kirkpatrick.get_planar_map(kirkpatrick.triangulation_data)

    # Extract triangles and plot them
    import matplotlib.pyplot as plt

    # plt.triplot([x[0] for x in kirkpatrick.all_points], [x[1] for x in kirkpatrick.all_points], kirkpatrick.base_triangulation, color='gray', alpha=0.5)
    # plt.show()

    def draw_planar_map(planar_map):
        fig, ax = plt.subplots()

        # Rysowanie wierzchołków
        for node in planar_map.iternodes():
            ax.plot(node.x, node.y, 'o', color='blue')  # Wierzchołki jako niebieskie kropki
            ax.text(node.x, node.y, f"{node.x, node.y}", fontsize=8, color='red')  # Indeksy wierzchołków

        # Rysowanie krawędzi
        for edge in planar_map.iteredges():
            start_node = edge.source  # Początkowy wierzchołek
            end_node = edge.target  # Końcowy wierzchołek
            x = [start_node.x, end_node.x]
            y = [start_node.y, end_node.y]
            ax.plot(x, y, 'k-', linewidth=1)  # Rysowanie krawędzi jako czarna linia

        # 3. Rysowanie twarzy (faces)
        # for face in planar_map.iterfaces():  # Iteracja po twarzach
        #     face_nodes = face.nodes  # Pobranie wierzchołków twarzy
        #     x_coords = [node.x for node in face_nodes] + [face_nodes[0].x]
        #     y_coords = [node.y for node in face_nodes] + [face_nodes[0].y]
        #     ax.fill(x_coords, y_coords, alpha=0.3, edgecolor='black', facecolor='green')  # Rysowanie twarzy

        # Ustawienia osi
        ax.set_aspect('equal', adjustable='datalim')
        plt.show()

    # Wywołanie funkcji
    draw_planar_map(planar_map)
