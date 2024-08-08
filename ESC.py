import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs, make_moons
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import random
from scipy.spatial import ConvexHull

class ClusteringAlgorithm:
    def __init__(self, grid_size, distance_threshold, population_size=100, generations=200, mutation_rate=0.1):
        self.grid_size = grid_size
        self.distance_threshold = distance_threshold
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    @staticmethod
    def generate_circle_data(n_samples=100, noise=0.05):
        X, _ = make_circles(n_samples=n_samples, factor=0.5, noise=noise)
        return X

    @staticmethod
    def generate_blobs_data(n_samples=100, centers=3, cluster_std=1.0):
        X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
        return X

    @staticmethod
    def generate_moons_data(n_samples=100, noise=0.1):
        X, _ = make_moons(n_samples=n_samples, noise=noise)
        return X

    @staticmethod
    def calculate_distance_matrix(data):
        return squareform(pdist(data))

    def grid_partitioning(self, data):
        min_vals = np.min(data, axis=0)
        grid_cells = {}

        for point in data:
            cell_coords = tuple(((point - min_vals) // self.grid_size).astype(int))
            if cell_coords not in grid_cells:
                grid_cells[cell_coords] = []
            grid_cells[cell_coords].append(point)

        return grid_cells

    @staticmethod
    def calculate_average_points(grid_cells):
        average_points = []
        for cell in grid_cells.values():
            average_points.append(np.mean(cell, axis=0))
        return np.array(average_points)

    class GeneticAlgorithmMST:
        def __init__(self, data, distance_threshold, population_size, generations, mutation_rate):
            self.data = data
            self.distance_matrix = ClusteringAlgorithm.calculate_distance_matrix(data)
            self.distance_threshold = distance_threshold
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate
            self.num_points = len(data)
            self.population = self.initialize_population()

        def initialize_population(self):
            population = []
            for _ in range(self.population_size):
                individual = self.create_random_spanning_forest()
                population.append(individual)
            return population

        def create_random_spanning_forest(self):
            graph = nx.Graph()
            graph.add_nodes_from(range(self.num_points))
            edges = [(i, j) for i in range(self.num_points) for j in range(i + 1, self.num_points) if self.distance_matrix[i, j] <= self.distance_threshold]
            graph.add_edges_from(random.sample(edges, min(len(edges), self.num_points - 1)))

            components = list(nx.connected_components(graph))
            return components

        def fitness(self, individual):
            return len(individual)

        def selection(self):
            selected = []
            for _ in range(self.population_size):
                i, j = random.sample(range(len(self.population)), 2)
                selected.append(min(self.population[i], self.population[j], key=self.fitness))
            return selected

        def crossover(self, parent1, parent2):
            graph1 = self.create_graph_from_components(parent1)
            graph2 = self.create_graph_from_components(parent2)
            graph_union = nx.compose(graph1, graph2)

            components = list(nx.connected_components(graph_union))
            child = []
            for component in components:
                subgraph = graph_union.subgraph(component)
                mst = nx.minimum_spanning_tree(subgraph)
                child.append(set(mst.nodes))
            return child

        def create_graph_from_components(self, components):
            graph = nx.Graph()
            for component in components:
                for node in component:
                    graph.add_node(node)
                for i in component:
                    for j in component:
                        if i != j and self.distance_matrix[i, j] <= self.distance_threshold:
                            graph.add_edge(i, j)
            return graph

        def mutate(self, individual):
            if random.random() < self.mutation_rate:
                graph = self.create_graph_from_components(individual)
                all_nodes = set(range(self.num_points))
                for component in individual:
                    if random.random() < self.mutation_rate:
                        if len(component) > 1:
                            node1, node2 = random.sample(component, 2)
                            if self.distance_matrix[node1, node2] <= self.distance_threshold:
                                graph.add_edge(node1, node2)
                        remaining_nodes = all_nodes - component
                        if remaining_nodes:
                            node1 = random.choice(list(component))
                            node2 = random.choice(list(remaining_nodes))
                            if self.distance_matrix[node1, node2] <= self.distance_threshold:
                                graph.add_edge(node1, node2)

                components = list(nx.connected_components(graph))
                return components
            return individual

        def evolve(self):
            for generation in range(self.generations):
                selected = self.selection()
                new_population = []
                for i in range(0, self.population_size, 2):
                    parent1, parent2 = selected[i], selected[i + 1]
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population.append(child)
                self.population = new_population

                best_individual = min(self.population, key=self.fitness)
                print(f"Generation {generation}: Minimum Trees = {self.fitness(best_individual)}")

            best_solution = min(self.population, key=self.fitness)
            return best_solution

        def cluster(self):
            best_solution = min(self.population, key=self.fitness)
            labels = np.full(self.num_points, -1, dtype=int)

            for cluster_id, component in enumerate(best_solution):
                for node in component:
                    labels[node] = cluster_id

            return labels, best_solution

    def evolutionary_spanning_clustering(self, data):
        grid_cells = self.grid_partitioning(data)
        average_points = self.calculate_average_points(grid_cells)
        
        ga_mst = self.GeneticAlgorithmMST(average_points, self.distance_threshold, self.population_size, self.generations, self.mutation_rate)
        solution = ga_mst.evolve()
        labels, best_solution = ga_mst.cluster()
        
        labels_original = np.empty(len(data), dtype=int)
        for i, point in enumerate(data):
            min_dist = float('inf')
            closest_label = -1
            for label, avg_point in enumerate(average_points):
                dist = np.linalg.norm(point - avg_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_label = labels[label]
            labels_original[i] = closest_label
        
        return labels_original, average_points, best_solution

    @staticmethod
    def plot_solution(data, average_points, solution, labels):
        plt.figure(figsize=(8, 8))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            component_points = data[labels == label]
            plt.scatter(component_points[:, 0], component_points[:, 1], s=10, label=f'Cluster {label}')
        
        for component in solution:
            points = average_points[list(component)]
            if len(points) > 2:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            else:
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        plt.plot(points[[i, j], 0], points[[i, j], 1], 'k-')
        plt.scatter(average_points[:, 0], average_points[:, 1], c='red', marker='*', s=100, label='Average Points')
        plt.title('ESC with Grid Partitioning')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()

