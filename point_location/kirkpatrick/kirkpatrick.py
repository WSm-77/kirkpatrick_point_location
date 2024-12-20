import networkx as nx
import triangle
import mapbox_earcut
from planegeometry.structures.planarmaps import PlanarMap, Point, Segment, Triangle
from functools import cmp_to_key
from computational_utils.utils import orient

from computational_utils.char_generator import unique_chars

class My_Triangle(Triangle):
    def __init__(self, *arguments, id):
        super().__init__(*arguments)
        self.id = id

    def __repr__(self):
        return f"{self.id}"

class Kirkpatrick:
    def __init__(self, input_points: list[tuple[float, float]], input_edges: list[tuple[int, int]]):
        # verify graph planarity
        self.assert_planarity(input_edges)

        self.input_points = input_points
        self.input_edges = input_edges

        embedding = self.get_embedding(input_points, input_edges)

        self.input_faces = self.filter_outer_face(self.get_faces(input_edges, embedding))

        self.bounding_triangle = self.get_bounding_triangle(input_points)

        self.all_points = self.input_points + self.bounding_triangle
        self.all_triangles = [] # do usunięcia

        self.points_to_idx = {tuple(point) : idx for idx, point in enumerate(self.all_points)}

        all_points_cnt = len(self.all_points)

        # last 3 points are points of bounding triangle
        self.bounding_triangle_points = {Point(*self.all_points[all_points_cnt - i - 1]) for i in range(3)}

        bounding_triangle_edges = [
            (all_points_cnt - 3, all_points_cnt - 2),
            (all_points_cnt - 2, all_points_cnt - 1),
            (all_points_cnt - 1, all_points_cnt - 3)
        ]

        tri_input = {
            'vertices': self.all_points,
            'segments': self.input_edges + bounding_triangle_edges,
        }

        self.triangulation_data = triangle.triangulate(tri_input, 'p')
        self.base_triangulation = self.triangulation_data['triangles']

        self.planar_map = self.get_planar_map(self.triangulation_data)

        # add input faces indices as leaf nodes to hierarchy tree/graph
        self.hierarchy_graph: dict[Triangle | int, list[Triangle] | None] = {i : None for i in range(len(self.input_faces))}

        self.initialize_hierarchy_graph()

    def get_independent_points_set(self):
        checked = set()
        independent_points = set()

        for point in self.planar_map.iterpoints():
            if point in checked:
                continue

            if point not in self.bounding_triangle_points and self.planar_map.degree(point) <= 8:
                independent_points.add(point)

                for neighbour in self.planar_map.iteradjacent(point):
                    checked.add(neighbour)

            checked.add(point)

        return independent_points

    def get_neighbour_triangles_list(self, point, neighbours_ordered_clockwise):
        neighbour_triangles_list = []

        prev_neighbour = neighbours_ordered_clockwise[-1]

        for next_neighbour in neighbours_ordered_clockwise:
            new_triangle = Triangle(prev_neighbour, next_neighbour, point)
            # new_triangle = My_Triangle(prev_neighbour, next_neighbour, point, id=unique_chars.pop(0))

            # for triangle in self.hierarchy_graph.keys():
            #     if isinstance(triangle, My_Triangle) and triangle == new_triangle:
            #         new_triangle = triangle
            #         break
            neighbour_triangles_list.append(new_triangle)
            self.all_triangles.append(new_triangle) # do usunięcia


            prev_neighbour = next_neighbour

        return neighbour_triangles_list

    def get_cmp_clockwise(self, point):
        return lambda p1, p2: (orient((point.x, point.y), (p1.x, p1.y), (p2.x, p2.y)))

    def retriangulate_hole(self, neighbours_ordered_clockwise):
        points_cnt = len(neighbours_ordered_clockwise)

        hole_points = [(point.x, point.y) for point in neighbours_ordered_clockwise]

        hole_indices = [points_cnt]

        triangles = mapbox_earcut.triangulate_float64(hole_points, hole_indices)

        return triangles.reshape(-1, 3)

    def get_triangles_from_triangulation(self, triangulation_data: dict, points_to_triangulate):
        triangles_list = []

        for a, b, c in triangulation_data:
            point_a = points_to_triangulate[a]
            point_b = points_to_triangulate[b]
            point_c = points_to_triangulate[c]

            new_triangle = Triangle(point_a, point_b, point_c)
            # new_triangle = My_Triangle(point_a, point_b, point_c, id=unique_chars.pop(0))

            triangles_list.append(new_triangle)
            self.all_triangles.append(new_triangle) # do usunięcia

        return triangles_list

    def add_missing_edges(self, new_triangles_list):
        for triangle in new_triangles_list:
            point_a = triangle.pt1
            point_b = triangle.pt2
            point_c = triangle.pt3

            edge1 = Segment(point_a, point_b)
            edge2 = Segment(point_b, point_c)
            edge3 = Segment(point_a, point_c)

            if not self.planar_map.has_edge(edge1):
                self.planar_map.add_edge(edge1)

            if not self.planar_map.has_edge(edge2):
                self.planar_map.add_edge(edge2)

            if not self.planar_map.has_edge(edge3):
                self.planar_map.add_edge(edge3)

    def process_hierarchy_of_triangles(self,prev_triangles: list[Triangle], new_triangles: list[Triangle]) -> None:

        def triangles_intersect(triangle_A:Triangle, triangle_B:Triangle) -> bool:
            for segment_A in triangle_A.itersegments():
                for segment_B in triangle_B.itersegments():
                    if segment_A.intersect(segment_B): return True
            return False

        for new_triangle in new_triangles:
            for prev_triangle in prev_triangles:
                if triangles_intersect(new_triangle, prev_triangle):
                    if new_triangle not in self.hierarchy_graph: self.hierarchy_graph[new_triangle] = []
                    self.hierarchy_graph[new_triangle].append(prev_triangle)

    def locate_point(self, point) -> int:
        if not isinstance(point, Point): point = Point(*point)

        top_of_dag = list(self.hierarchy_graph.keys())[-1]
        if not top_of_dag.__contains__(point):
            return #point is in exterior face

        triangle_containing_point = top_of_dag
        
        while True:
            flag = False
            for child in self.hierarchy_graph[triangle_containing_point]:
                if isinstance(child, tuple): return child
                # if isinstance(child, tuple): return self.face_to_points(child[1])
                if child.__contains__(point):
                    triangle_containing_point = child
                    flag = True
                    break
            if not flag: break

        return None

    def preprocess(self):
        input_points_cnt = len(self.input_points)

        while input_points_cnt > 0:
            independent_points_set = self.get_independent_points_set()

            # print(f"\n\n\n######### NEXT ITTERATION #########\n\n\n")


            for independent_point in independent_points_set:
                neighbours_ordered_clockwise = [point for point in self.planar_map.iteradjacent(independent_point)]
                neighbours_ordered_clockwise.sort(key = cmp_to_key(self.get_cmp_clockwise(independent_point)))

                neighbour_triangles_list = self.get_neighbour_triangles_list(independent_point, neighbours_ordered_clockwise)

                # print(f"#### independent point ####\n\n")

                # print(independent_point)

                # print("\n\n")

                # print(f"#### neighbours clockwise ####\n\n")

                # print(list(map(lambda p: (float(p.x), float(p.y)), neighbours_ordered_clockwise)))

                # print("\n\n")

                # print(*neighbour_triangles_list, sep="\n")

                self.planar_map.del_node(independent_point)

                triangulation_data = self.retriangulate_hole(neighbours_ordered_clockwise)

                # print(f"#### triangles ####\n\n")

                # print(triangulation_data)

                # print("\n\n")

                new_triangles_list: list[Triangle] = self.get_triangles_from_triangulation(triangulation_data, neighbours_ordered_clockwise)

                self.add_missing_edges(new_triangles_list)
                
                self.process_hierarchy_of_triangles(neighbour_triangles_list, new_triangles_list)

                # break

            input_points_cnt -= len(independent_points_set)

            # draw_planar_map(self.planar_map)        # TODO: remove (currently for testing purpose)

    def assert_planarity(self, input_edges):
        planar_subdivision = nx.Graph()
        planar_subdivision.add_edges_from(input_edges)

        is_planar, _ = nx.check_planarity(planar_subdivision)

        if not is_planar:
            raise ValueError("This subdivision is not planar!!!")

    def initialize_hierarchy_graph(self):
        faces_sets = [({edge[0] for edge in face}, face) for face in self.input_faces]

        for a, b, c in self.base_triangulation:
            point_a = self.idx_to_point(a)
            point_b = self.idx_to_point(b)
            point_c = self.idx_to_point(c)

            triangle_object = Triangle(point_a, point_b, point_c)
            # triangle_object = My_Triangle(point_a, point_b, point_c, id=unique_chars.pop(0))

            if triangle_object not in self.hierarchy_graph:
                self.hierarchy_graph[triangle_object] = []
                self.all_triangles.append(triangle_object) # do usunięcia

            for face_idx, face in enumerate(faces_sets):
                face_set, face_raw = face
                # check if current triangle is included in current face
                if (a in face_set) and (b in face_set) and (c in face_set):
                    self.hierarchy_graph[triangle_object].append((face_idx, face_raw))
                    # this triangle can be included only by one face, so we can break
                    break

    def filter_outer_face(self, input_faces):
        filtered_faces = []

        for face in input_faces:
            p1 = face[0][0]
            p2 = face[1][0]
            p3 = face[2][0]

            if orient(self.input_points[p1], self.input_points[p2], self.input_points[p3]) == -1:
                continue

            filtered_faces.append(face)

        return filtered_faces

    def idx_to_point(self, idx: int) -> Point:
        return Point(*self.all_points[idx])

    def point_to_idx(self, point: tuple[float, float]) -> int:
        return self.points_to_idx[point]
    
    def face_to_points(self, face:list[tuple[int,int]]):
        result = set()
        for idx_1, idx_2 in face:
            result.add(self.idx_to_point(idx_1))
            result.add(self.idx_to_point(idx_2))
        result = list(result)
        result.sort(key = cmp_to_key(self.get_cmp_clockwise(result[0]))) # problem <- układ punktów jest zły, bo nie jest sortowny od wewnątrz ściany
        return result

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

    def get_embedding(self, points, edges):
        def get_cmp_clockwise_for_embedding(vertex):
            return lambda v1, v2: (orient(vertex_to_point[vertex], vertex_to_point[v1], vertex_to_point[v2]))

        graph: dict[int, list] = {}
        vertex_to_point = {}

        for v, point in enumerate(points):
            graph[v] = []
            vertex_to_point[v] = point

        for vertex, neighbour in edges:
            graph[vertex].append(neighbour)
            graph[neighbour].append(vertex)

        for vertex in graph.keys():
            graph[vertex].sort(key = cmp_to_key(get_cmp_clockwise_for_embedding(vertex)))

        return graph


#### TESTING ####

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

    ax.set_aspect('equal', adjustable='datalim')
    plt.show()

def draw_initial_faces(vertices, edges):
    plt.figure()
    
    # Draw edges
    for edge in edges:
        start = vertices[edge[0]]
        end = vertices[edge[1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-')  # Draw edge
    
    # Draw vertices
    for vertex in vertices:
        plt.plot(vertex[0], vertex[1], 'ro')  # Draw vertex
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    import json
    import os
    import matplotlib.pyplot as plt

    test_no = 1
    test_dir = "point_location/kirkpatrick/test_input"
    test_name = f"test{test_no}.json"

    with open(os.path.join(test_dir, test_name), 'r') as file:
        data = json.load(file)

    vertices = data["vertices"]
    segments = data["segments"]

    # kirkpatrick = Kirkpatrick(list(map(tuple, list(vertices))), list(map(tuple, list(segments))))
    kirkpatrick = Kirkpatrick(vertices, segments)
    # print(kirkpatrick.all_points)
    # print(kirkpatrick.bounding_triangle_points)
    # print(kirkpatrick.base_triangulation)
    # print(*kirkpatrick.input_faces, sep="\n")
    draw_initial_faces(vertices,segments)
    draw_planar_map(kirkpatrick.planar_map)

    kirkpatrick.preprocess()
    
    if test_no == 1:
        # TEST 1
        points_A = data["points_A"]
        for point in points_A:
            print(f'point: {point} is inside {kirkpatrick.locate_point(point)}')

    if test_no == 2:
        # TEST 2
        points_A = data["points_A"]
        for point in points_A:
            print(f'point: {point} is inside {kirkpatrick.locate_point(point)}')

    if test_no == 3:
        # TEST 3
        # points that should be in first face (test 3)
        points_A = data["points_A"]

        # points that should be in the second face (test 3)
        points_B = data["points_B"]

        # points that are outside of input faces
        points_outside = data["points_outside"]

        # points on the edge
        edge_points = data["edge_points"]

        print("TESTING POINTS IN FIRST FACE")
        for point in points_A:
            print(f'point: {point} is inside {kirkpatrick.locate_point(point)}')

        print("\nTESTING POINTS IN SECOND FACE")
        for point in points_B:
            print(f'point: {point} is inside {kirkpatrick.locate_point(point)}')

        print("\nTESTING POINTS OUTSIDE")
        for point in points_outside:
            print(f'point: {point} is inside {kirkpatrick.locate_point(point)}')
        
        print("\nTESTING EDGE POINTS")
        for point in edge_points:
            print(f'point: {point} is inside {kirkpatrick.locate_point(point)}')