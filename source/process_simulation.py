import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os


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

        self.next_nearest = []
        for p in range(len(self.xyz)):
            self.next_nearest.append(set({}))
            for neighbour in list(self.connectivity[p]):
                self.next_nearest[-1].update(self.connectivity[neighbour])


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
filename = 'data/Mg2SiO4_2500K_1bar.dat'

sim = Simulation(filename=filename)


coordination = {'Mg': [], 'Si': [], 'O': []}
O_coordination = {'Mg': [], 'Si': []}

next_nearest_neighbours = []
elements = []
for i_conf in range(14000, 15000):
    sim.configurations[i_conf].process(cutoff_scaling={'Mg': 1., 'Si': 1., 'O': 1.})
    for i in range(0,96):
        coordination[sim.element_list[i%sim.total_atoms]].append(len(sim.configurations[i_conf].connectivity[i]))

        if sim.element_list[i%sim.total_atoms] == 'O':
            O_coordination['Mg'].append(0)
            O_coordination['Si'].append(0)

            for p_connector in sim.configurations[i_conf].connectivity[i]:
                if sim.element_list[p_connector%sim.total_atoms] == 'Mg':
                    O_coordination['Mg'][-1] += 1
                if sim.element_list[p_connector%sim.total_atoms] == 'Si':
                    O_coordination['Si'][-1] += 1

        next_nearest_atoms = sim.configurations[i_conf].next_nearest[i]
        elements.append(sim.element_list[i%sim.total_atoms])
        next_nearest_neighbours.append({'Mg': 0., 'Si': 0., 'O': 0.})
        for n in next_nearest_atoms:
            next_nearest_neighbours[-1][sim.element_list[n%sim.total_atoms]] += 1.


c_max = 20
n_next_nearest = {'Mg': {'Mg': [0 for i in range(c_max)],
                       'Si': [0 for i in range(c_max)],
                       'cation': [0 for i in range(c_max)]},
                'Si': {'Mg': [0 for i in range(c_max)],
                       'Si': [0 for i in range(c_max)],
                       'cation': [0 for i in range(c_max)]}}

f_next_nearest = {'Mg': {'Mg': [[] for i in range(c_max)],
                       'Si': [[] for i in range(c_max)]},
                'Si': {'Mg': [[] for i in range(c_max)],
                       'Si': [[] for i in range(c_max)]}}
for i in range(len(elements)):
    n = next_nearest_neighbours[i]
    if elements[i] == 'Mg' or elements[i] == 'Si':

        if int(n['Mg']) < c_max:
            n_next_nearest[elements[i]]['Mg'][int(n['Mg'])] += 1

        if int(n['Si']) < c_max:
            n_next_nearest[elements[i]]['Si'][int(n['Si'])] += 1

        if int(n['Mg'] + n['Si']) < c_max:
            n_next_nearest[elements[i]]['cation'][int(n['Mg'] + n['Si'])] += 1

        if int(n['Mg'] + n['Si'] + n['O']) < c_max:
            f_next_nearest[elements[i]]['Mg'][int(n['Mg'] + n['Si'] + n['O'])].append(n['Mg']/(n['Mg'] + n['Si'] + n['O']))
            f_next_nearest[elements[i]]['Si'][int(n['Mg'] + n['Si'] + n['O'])].append(n['Si']/(n['Mg'] + n['Si'] + n['O']))



fig = plt.figure(figsize=(15,10))
ax = [fig.add_subplot(2,2,i) for i in range(1, 5)]


#
for a1 in ['Mg', 'Si']:
    for a2 in ['Mg', 'Si']:
        ax[0].plot(range(c_max),
                   np.array(map(float, n_next_nearest[a1][a2]))/sum(n_next_nearest[a1][a2]),
                   label='{0}-{1}'.format(a1, a2))
    for a2 in ['cation']:
        ax[1].plot(range(c_max),
                   np.array(map(float, n_next_nearest[a1][a2]))/sum(n_next_nearest[a1][a2]),
                   label='{0}-{1}'.format(a1, a2))


ax[0].set_xlim(0,c_max)
ax[0].set_xticks(range(0, c_max, 2))
ax[0].set_xlabel('coordination')
ax[0].set_ylabel('atom fraction')
ax[0].legend()


ax[1].set_xlim(0,c_max)
ax[1].set_xticks(range(0, c_max, 2))
ax[1].set_xlabel('coordination')
ax[1].set_ylabel('atom fraction')
ax[1].legend()

# Nearest neighbour coordination
coordinations = range(c_max+1)
for atom in ['Mg', 'Si', 'O']:
    xs, cs = np.unique(coordination[atom], return_counts=True)
    counts = np.zeros_like(coordinations).astype(float)
    counts[xs] = map(float, cs)
    if atom == 'O':
        ax[2].plot(coordinations, counts/sum(counts), label='{0}-cation'.format(atom))
    else:
        ax[2].plot(coordinations, counts/sum(counts), label='{0}-O'.format(atom))

for atom in ['Mg', 'Si']:
    xs, cs = np.unique(O_coordination[atom], return_counts=True)
    counts = np.zeros_like(coordinations).astype(float)
    counts[xs] = map(float, cs)
    ax[2].plot(coordinations, counts/sum(counts), label='O-{0}'.format(atom))

ax[2].set_xlim(0,c_max/2.)
ax[2].set_xticks(range(0, c_max/2, 2))
ax[2].set_xlabel('coordination')
ax[2].set_ylabel('atom fraction')
ax[2].legend()


# Coordination fractions
ax[3].plot(range(c_max), [np.average(map(float, fs)) for fs in f_next_nearest['Mg']['Mg']], label='Mg-Mg')
ax[3].plot(range(c_max), [np.average(map(float, fs)) for fs in f_next_nearest['Si']['Mg']], label='Si-Mg')


ax[3].set_xlim(0,c_max)
ax[3].set_xticks(range(0, c_max, 2))
ax[3].set_ylim(0,1)

ax[3].set_xlabel('Coordination number A+B around atom A')
ax[3].set_ylabel('Fraction atom B around atom A')

ax[3].legend()


base = os.path.basename(filename)
basename = os.path.splitext(base)[0]
fig.savefig('output/'+basename+'_coordination_fractions.pdf')
plt.show()

