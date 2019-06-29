import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay



class Configuration(object):
    def __init__(self, unscaled_data, box_dimensions, elements, n_atoms, name='unnamed configuration'):
        #print('Loading {0}'.format(name))
        self.name = name
        self.unscaled_data = unscaled_data
        self.data = unscaled_data*box_dimensions
        self.box_dimensions = box_dimensions
        self.elements = elements
        self.n_atoms = n_atoms
        self.total_atoms = sum(n_atoms)
        self.element_list = [item for sublist in [[e]*n_atoms[i]
                                                  for i, e in enumerate(elements)]
                             for item in sublist]
        
    def process(self, cutoff_scaling):
        print('Processing {0}'.format(self.name))
        d = self.unscaled_data        
        self.xyz = np.vstack((d,
                              d + [-1., 0., 0.], d + [1., 0., 0.],
                              d + [0., -1., 0.], d + [0., 1., 0.],
                              d + [0., 0., -1.], d + [0., 0., 1.],
                              d + [-1., -1., 0.], d + [1., 1., 0.],
                              d + [-1., 1., 0.], d + [1., -1., 0.],
                              d + [0., -1., -1.], d + [0., 1., 1.],
                              d + [0., -1., 1.], d + [0., 1., -1.],
                              d + [-1., 0., -1.], d + [1., 0., 1.],
                              d + [-1., 0., 1.], d + [1., 0., -1.],
                              d + [-1., -1., -1.], d + [1., 1., 1.],
                              d + [-1., -1., 1.], d + [1., 1., -1.],
                              d + [-1., 1., 1.], d + [1., -1., -1.],
                              d + [-1., 1., -1.], d + [1., -1., 1.],))*self.box_dimensions
        
        self.triangulation = Delaunay(self.xyz)
        
        self.connectivity = [set([]) for i in self.xyz]
        for simplex in self.triangulation.simplices:
            for p in simplex:
                self.connectivity[p].update(simplex)

        # remove self-connections and disallowed connections
        for p in range(len(self.xyz)):
            self.connectivity[p].remove(p)
    
            # remove disallowed connections
            removed_lengths = []
            for p1 in list(self.connectivity[p]):
                if self.element_list[p%self.total_atoms] == 'O' and self.element_list[p1%self.total_atoms] == 'O': # remove O-O
                    self.connectivity[p].remove(p1)
                    removed_lengths.append(np.linalg.norm(self.xyz[p] - self.xyz[p1]))

                if self.element_list[p%self.total_atoms] != 'O' and self.element_list[p1%self.total_atoms] != 'O': # remove Mg-Mg, Mg-Si, Si-Si
                    self.connectivity[p].remove(p1)
                    removed_lengths.append(np.linalg.norm(self.xyz[p] - self.xyz[p1]))
                
            # remove allowed connections that are longer than the shortest disallowed connection
            # IDEA: if necessary (for very anisotropic sites?),
            #       could remove points outside the polyhedron
            #       defined by the removed points.
            #       Problem: sometimes v few removed points!
            n_removed_lengths = len(removed_lengths)
            if n_removed_lengths > 0:
                #print(n_removed_lengths)
                cutoff = min(removed_lengths)*cutoff_scaling[self.element_list[p%self.total_atoms]]

                for p1 in list(self.connectivity[p]):
                    length = np.linalg.norm(self.xyz[p] - self.xyz[p1])
                    if length > cutoff:
                        self.connectivity[p].remove(p1)
                        

        
class Simulation(object):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            datastream = f.read()
            datalines = [line.strip().split()
                         for line in datastream.split('\n')]

            self.name = datalines[0][0]
            shrug = datalines[1][0] # I don't know what this is

            self.box_dimensions = np.array([map(float, line) for line in datalines[2:5]])

            for i in range(2):
                for j in range(i+1, 2):
                    assert self.box_dimensions[i,j] == 0
                    assert self.box_dimensions[j,i] == 0

            self.box_dimensions = np.diag(self.box_dimensions)
            self.elements = datalines[5]
            self.n_atoms = map(int, datalines[6])
            self.total_atoms = sum(self.n_atoms)

            cation_fraction = np.array(map(float, self.n_atoms))
            cation_fraction /= sum(cation_fraction[0:2])

            element_starts = [0]
            element_starts.extend(np.cumsum(map(int, datalines[6])))

            atom_ranges = {}
            for i, element in enumerate(self.elements):
                atom_ranges[element] = [element_starts[i], element_starts[i+1]]
    
            print(self.elements)
            print(self.box_dimensions)
            print(self.n_atoms)
            print(self.total_atoms)

            
            self.element_list = [item for sublist in [[e]*self.n_atoms[i]
                                                      for i, e in enumerate(self.elements)]
                                 for item in sublist]
            
            self.n_configurations = (len(datalines) - 7)/(self.total_atoms+1)
            print('Number of configurations: {0}'.format(self.n_configurations))
            self.configurations = []
            for i in range(self.n_configurations):
                unscaled_data = np.array([map(float, line)
                                          for line
                                          in datalines[8 + i*(self.total_atoms+1):7 +
                                                       (i+1)*(self.total_atoms+1)]])
                self.configurations.append(Configuration(unscaled_data = unscaled_data,
                                                         box_dimensions = self.box_dimensions,
                                                         elements = self.elements,
                                                         n_atoms = self.n_atoms,
                                                         name = 'Configuration {0}'.format(i)))




filename = 'data/MgSiO3_3000K_1bar.dat'
#filename = 'data/MgSiO3_5000K_1bar.dat'
#filename = 'data/Mg2SiO4_2500K_1bar.dat'

sim = Simulation(filename=filename)


coordination = {'Mg': [], 'Si': [], 'O': []}

for i_conf in range(14900, 15000):
    sim.configurations[i_conf].process(cutoff_scaling={'Mg': 1., 'Si': 1., 'O': 1.})
    for i in range(0,96):
        coordination[sim.element_list[i%sim.total_atoms]].append(len(sim.configurations[i_conf].connectivity[i]))


bins = np.linspace(0.5, 12.5, 13)
for atom in ['Mg', 'Si', 'O']:
    plt.hist(coordination[atom], bins = bins, histtype='step', label=atom)
plt.legend()
plt.show()

