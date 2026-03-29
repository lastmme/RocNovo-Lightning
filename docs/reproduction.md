# Model Reproduction

We provide the training and inference configuration files for each dataset in the yamls directory of this repository. To reproduce the experimental results reported in benchmarks, you simply need to execute the training process using the corresponding configuration file. Additionally, the saved model checkpoints are named with their validation set performance, which can serve as a reference during your reproduction.

Training the model on the MassIVE-KB dataset takes approximately 2 days for the CLIP stage, with 3 days and 14 hours for the de novo stage.

Training the models across all three subsets of the NovoBench dataset takes a total of about 2 days. (Note: This duration is largely due to the use of a small batch size during the de novo training phase. You can accelerate the training speed without compromising performance by adopting a larger batch size along with appropriate hyperparameter tuning.)