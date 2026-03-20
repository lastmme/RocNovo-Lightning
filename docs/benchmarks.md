# Benchmarks

## Inference Speed

All inference experiments were conducted on a single NVIDIA RTX 4090 GPU.

The nine Species V1 dataset is the full-scale dataset, not the subsampled 10k `.mgf` subset.

The Nine Species V2 dataset is identical to the dataset used in the Casanovo V2 paper. However, we filtered out a subset of the data that exhibited excessively large deviations between the observed precursor ion m/z and the theoretical m/z calculated from the sequence annotations. Detailed data counts for each species post-filtering can be found in the appendix of our paper.

### Average Throughput (Spectra/s)

*The figure below illustrates the average speed across different search strategies (Greedy Search vs. Beam Search).*

#### Nine Species V1
![](./figures/benchmarks/nine_species_v1_avg_speed.png)

#### Nine Species V2
![](./figures/benchmarks/nine_species_v2_avg_speed.png)


### Speed per Species

*A detailed breakdown of the inference speed across the nine individual species.*

#### Nine Species V1
![](./figures/benchmarks/nine_species_v1_species_speed.png)

#### Nine Species V2
![](./figures/benchmarks/nine_species_v2_species_speed.png)
