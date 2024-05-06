CONFIG = {
    "SEED": 42,
    "DATA": {
        "train_data_dir": "./data/scenario1/train",
        "test_data_dir": "./data/scenario1/test",
        "batch_size": 32,
        "shape": (128, 128),
    },
    "TRAIN": {
        "num_epochs": 100,
        "warmup_epochs": 20,
        "lr": 0.001,
        "loss_weight": 0.5,
        "clip_grad": 5,
        "alpha_lr": 0.01,
        "weight_lr": 0.001,
        "weight_decay": 2e-4,
        "sample_weight_lr": 1e-5,
    },
    "GPU": [2,3]
}
