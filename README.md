# Evolutionary-Spanning-Clustering-ESC-
A novel clustering algorithm that uses GA and spanning forest concepts for clustering.


ESC effectively handles complex clustering tasks where traditional methods may fail by forming naturally separated clusters based on the inherent structure of the data, provided that a reasonable distance threshold is given.

For large datasets, a grid-based approach should be taken to speed up the process. See the related notebook attached for a grid-based approach.

![image](https://github.com/user-attachments/assets/a0a453e0-d3e5-44c7-aab2-59fc90f6c71d)

## Key Features

- **Genetic Algorithm-Based Clustering:** Utilizes genetic algorithms to optimize the clustering process.
- **Adaptive Clustering:** Automatically adapts to the natural structure of data, without imposing artificial boundaries.
- **Effective for Complex Datasets:** Proven performance on synthetic datasets, including circles, blobs, and moons.
- **Extremely fast when using the grid-based approach.**

### Note: selecting a reasonable threshold and grid size is important for a good performance, nevertheless, if this information is not available, it can be found during the training/validation phase.

### ESC Pseudocode

```plaintext
Algorithm: Evolutionary Spanning Clustering (ESC)

Input: Dataset X, distance threshold D, 
       population size P, number of generations G, 
       mutation rate μ

Output: Optimal spanning forest F*

1. Initialize population 𝒫 with P random spanning forests
2. For generation g = 1 to G:
   3. For each individual F in 𝒫:
      4. Compute fitness f(F) = k(F)
   5. Select parents from 𝒫 based on fitness
   6. Apply crossover to parents to produce offspring
   7. Apply mutation to offspring with probability μ
   8. Replace worst individuals in 𝒫 with offspring
9. Return F* = argmin_{F in 𝒫} f(F)

