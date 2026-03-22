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

## Metrics

Isoleucine (I) and Leucine (L) are not distinguished and are treated as equivalent.

In mass spectrometry *de novo* sequencing, standard evaluation metrics include Amino Acid Precision, Amino Acid Recall, and Peptide Recall (which is equivalent to Peptide Precision). These are typically defined by requiring both the single-step residue mass difference and the cumulative mass difference to be within 0.1 Da and 0.5 Da.

However, cases exist where the masses are similar but the actual residues differ, such as `D` vs. `N+0.984` and `E` vs. `Q+0.984`. Therefore, we introduced strict character-level metrics alongside the traditional mass-based ones. A prediction is only considered fully successful if the predicted sequence exactly matches the ground truth at the character level.

### NovoBench

This dataset consists of three subsets: HC PT, Nine Species, and Seven Species. Each subset is already pre-split into training, validation, and test sets. We have additionally included PTM Precision, PTM Recall, and a peptide-level AUC metric for comprehensive evaluation.

Please note that during data processing, we found discrepancies in the HC PT dataset where the precursor ion m/z does not match the theoretical m/z calculated from the annotated peptide. We thank the authors of RefineNovo for bringing this to our attention. However, since other baseline methods were trained on this **unfiltered data**, we still used it for training and testing in the results below to ensure a fair comparison.

Furthermore, we observed that using a large batch size on this dataset under the current settings leads to poor convergence. We suspect this might be caused by the data quality issues. (While this might be mitigated through careful hyperparameter tuning, we plan to add an Optuna-based hyperparameter search script for small datasets to our tutorials in the future). The results reported below were obtained by training the model on two NVIDIA RTX 4090 GPUs with a per-GPU batch size of 32. Please refer to the specific configuration files for complete training details.

**Correction Note:** Additionally, a clarification regarding the PTM metrics: Since all `C` residues were uniformly replaced with `C+57.021`, our original manuscript mistakenly included this fixed modification in the PTM metric calculations for the HC PT dataset. The results presented below reflect the corrected metrics. The script for the updated calculation can be found at [06_metric_calculation.ipynb](../tutorials/06_metric_calculation.ipynb).

| Dataset / Species | AA Precision | AA Recall | Peptide Precision | Peptide Recall | PTM Precision | PTM Recall | Full Accuracy | Curve AUC |
| :---------------- | :----------: | :-------: | :---------------: | :------------: | :-----------: | :--------: | :-----------: | :-------: |
| **HC PT**         |    0.670     |   0.670   |       0.513       |     0.513      |     0.743     |   0.780    |     0.512     |   0.472   |
| **Nine Species**  |    0.832     |   0.833   |       0.665       |     0.665      |     0.835     |   0.765    |     0.654     |   0.633   |
| **Seven Species** |    0.568     |   0.571   |       0.368       |     0.368      |     0.544     |   0.514    |     0.357     |   0.310   |

> Full Accuracy denotes the strict character-level sequence matching accuracy.