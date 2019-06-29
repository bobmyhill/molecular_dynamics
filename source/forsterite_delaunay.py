import numpy as np
from scipy.spatial import Delaunay

unit_cell_dimensions = np.array([4.75030, 10.18700,  5.97710])
xyz = np.loadtxt('test_data/forsterite.xyz')[:,0:3] * unit_cell_dimensions


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


element_mapping = ['Si', 'Mg', 'O']
colors = ['red', 'blue', 'green']
cutoff_scaling = [1., 1., 1.] # if > 1.05, Mg becomes > octahedrally coordinated
element = np.array([0, 0, 0, 0,
                    1, 1, 1, 1,
                    1, 1, 1, 1,
                    2, 2, 2, 2,
                    2, 2, 2, 2,
                    2, 2, 2, 2,
                    2, 2, 2, 2])

element = np.hstack([element]*27)
xyz = np.vstack((xyz,
                 xyz + [-1., 0., 0.]*unit_cell_dimensions, xyz + [1., 0., 0.]*unit_cell_dimensions,
                 xyz + [0., -1., 0.]*unit_cell_dimensions, xyz + [0., 1., 0.]*unit_cell_dimensions,
                 xyz + [0., 0., -1.]*unit_cell_dimensions, xyz + [0., 0., 1.]*unit_cell_dimensions,
                 xyz + [-1., -1., 0.]*unit_cell_dimensions, xyz + [1., 1., 0.]*unit_cell_dimensions,
                 xyz + [-1., 1., 0.]*unit_cell_dimensions, xyz + [1., -1., 0.]*unit_cell_dimensions,
                 xyz + [0., -1., -1.]*unit_cell_dimensions, xyz + [0., 1., 1.]*unit_cell_dimensions,
                 xyz + [0., -1., 1.]*unit_cell_dimensions, xyz + [0., 1., -1.]*unit_cell_dimensions,
                 xyz + [-1., 0., -1.]*unit_cell_dimensions, xyz + [1., 0., 1.]*unit_cell_dimensions,
                 xyz + [-1., 0., 1.]*unit_cell_dimensions, xyz + [1., 0., -1.]*unit_cell_dimensions,
                 xyz + [-1., -1., -1.]*unit_cell_dimensions, xyz + [1., 1., 1.]*unit_cell_dimensions,
                 xyz + [-1., -1., 1.]*unit_cell_dimensions, xyz + [1., 1., -1.]*unit_cell_dimensions,
                 xyz + [-1., 1., 1.]*unit_cell_dimensions, xyz + [1., -1., -1.]*unit_cell_dimensions,
                 xyz + [-1., 1., -1.]*unit_cell_dimensions, xyz + [1., -1., 1.]*unit_cell_dimensions,))

tri = Delaunay(xyz)
connectivity = [set([]) for i in xyz]
for simplex in tri.simplices:
    for p in simplex:
        connectivity[p].update(simplex)

# remove self-connections and disallowed connections
for p in range(len(xyz)):
    connectivity[p].remove(p)
    
    # remove disallowed connections
    removed_lengths = []
    for p1 in list(connectivity[p]):
        if element[p] == 2 and element[p1] == 2: # remove O-O
            connectivity[p].remove(p1)
            removed_lengths.append(np.linalg.norm(xyz[p] - xyz[p1]))

        if element[p] != 2 and element[p1] != 2: # remove Mg-Mg, Mg-Si, Si-Si
            connectivity[p].remove(p1)
            removed_lengths.append(np.linalg.norm(xyz[p] - xyz[p1]))

    # remove allowed connections that are longer than the shortest disallowed connection
    # IDEA: if necessary (for very anisotropic sites?),
    #       could remove points outside the polyhedron
    #       defined by the removed points.
    #       Problem: sometimes v few removed points!
    n_removed_lengths = len(removed_lengths)
    if n_removed_lengths > 0:
        #print(n_removed_lengths)
        cutoff = min(removed_lengths)*cutoff_scaling[element[p]]

        for p1 in list(connectivity[p]):
            length = np.linalg.norm(xyz[p] - xyz[p1])
            if length > cutoff:
                connectivity[p].remove(p1)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')


for i in range(56):
    x, y, z = xyz[i]
    ax.scatter(x, y, z, color=colors[element[i]], s=200) 


for k in [0, 28]:
    for i in range(0+k,4+k):
        for j in connectivity[i]:
            ax.plot([xyz[i][0], xyz[j][0]],
                    [xyz[i][1], xyz[j][1]],
                    [xyz[i][2], xyz[j][2]], color='red')
    for i in range(4+k,12+k):
        for j in connectivity[i]:
            ax.plot([xyz[i][0], xyz[j][0]],
                    [xyz[i][1], xyz[j][1]],
                    [xyz[i][2], xyz[j][2]], color='blue')


coordination = []
for i in range(0,28):
    coordination.append(len(connectivity[i]))
print(coordination)
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 10)
ax.set_zlim(-4, 8)
fig.savefig('output/forsterite.pdf')
plt.show()
