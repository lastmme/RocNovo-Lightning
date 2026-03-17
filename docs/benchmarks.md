# Benchmarks

## Inference Speed

All inference experiments were conducted on a single NVIDIA RTX 4090 GPU.

The dataset is the nine species V1.(the full-scale dataset, not the subsampled 10k `.mgf` subset)

### Average Throughput (Spectra/s)

*The figure below illustrates the average speed across different search strategies (Greedy Search vs. Beam Search).*

![](./figures/benchmarks/nine_species_v1_avg_speed.png)

### Speed per Species

*A detailed breakdown of the inference speed across the nine individual species.*

![](./figures/benchmarks/nine_species_v1_species_speed.png)