from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np


# points = np.array([(1,1), (2, 1), (2,2), (1.5, 3), (1, 2)])
points = np.array([(1,1), (2, 1), (2,2), (1.5, 1.5), (1, 2)])

# plt.scatter([point[0] for point in points], [point[1] for point in points])
plt.scatter(points[:, 0], points[:, 1])

delaunay = Delaunay(points)

print(delaunay.simplices)

for a, b, c in delaunay.simplices:
    point_a = points[a]
    point_b = points[b]
    point_c = points[c]
    plt.plot([point_a[0], point_b[0], point_c[0], point_a[0]], [point_a[1], point_b[1], point_c[1], point_a[1]])

plt.show()
