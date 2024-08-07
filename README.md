# Evolutionary-Spanning-Clustering-ESC-
A novel clustering algorithm that works great on all sorts of data distributions


ESC effectively handles complex clustering tasks where traditional methods may fail by forming naturally separated clusters based on the inherent structure of the data.

## Key Features

- **Genetic Algorithm-Based Clustering:** Utilizes genetic algorithms to optimize the clustering process.
- **Adaptive Clustering:** Automatically adapts to the natural structure of data, without imposing artificial boundaries.
- **Effective for Complex Datasets:** Proven performance on synthetic datasets, including circles, blobs, and moons.

Algorithm: Evolutionary Spanning Clustering (ESC)

Input: Dataset \( \mathbf{X} \), distance threshold \( D \), population size \( P \), number of generations \( G \), mutation rate \( \mu \)

Output: Optimal spanning forest \( F^* \)

1. Initialize population \( \mathcal{P} \) with \( P \) random spanning forests
2. For generation \( g = 1 \) to \( G \):
   3. For each individual \( F \in \mathcal{P} \):
      4. Compute fitness \( f(F) = k(F) \)
   5. Select parents from \( \mathcal{P} \) based on fitness
   6. Apply crossover to parents to produce offspring
   7. Apply mutation to offspring with probability \( \mu \)
   8. Replace worst individuals in \( \mathcal{P} \) with offspring
9. Return \( F^* \gets \arg\min_{F \in \mathcal{P}} f(F) \)

A sample of this algorithm is attached as a notebook, in which you can find experimental results. 
