import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter Configuration")

    # Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=600, help="Vocabulary size")
    parser.add_argument(
        "--max_seq_length", type=int, default=1000, help="Maximum sequence length"
    )
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=256, help="Feedforward network dimension"
    )
    parser.add_argument("--feature_dim", type=int, default=1, help="Feature dimension")
    parser.add_argument(
        "--AP_num", type=int, default=137, help="Number of APs (Access Points)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs for pretraining"
    )
    parser.add_argument(
        "--num_per_label_for_train", type=int, default=100, help="Number of sample per labels for pretraining"
    )
    parser.add_argument(
        "--task_epochs",
        type=int,
        default=400,
        help="Number of epochs for task-specific training",
    )
    parser.add_argument(
        "--lr_for_pretrain",
        type=float,
        default=0.005,
        help="Learning rate for pretraining",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="Learning rate decay rate"
    )
    parser.add_argument(
        "--is_sort", action="store_true", help="Whether to sort the data"
    )
    parser.add_argument(
        "--test_masking_rate", type=float, default=0.0, help="Rate of test masking"
    )
    parser.add_argument(
        "--test_fault_rate", type=float, default=0.0, help="Rate of test faults"
    )
    parser.add_argument(
        "--reference_points_ratio",
        type=float,
        default=0.1,
        help="Ratio of reference points",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gpt2", "transformer", "NN"],
        default="gpt2",
        help="Model type",
    )
    parser.add_argument(
        "--output_layer",
        type=str,
        choices=["RF", "NN"],
        default="NN",
        help="Model type",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hamada", "hamada_cell", "uji"],
        default="hamada",
        help="Dataset",
    )
    parser.add_argument(
        "--pre_train_method",
        type=str,
        choices=["rnc", "bert", "end-to-end", "label_aug_rnc"],
        default="rnc",
        help="Pretrain method",
    )
    parser.add_argument(
        "--model_setting",
        type=str,
        choices=["pre_trained", "random"],
        default="pre_trained",
        help="Model initialization setting",
    )
    parser.add_argument(
        "--aug_setting",
        type=str,
        choices=["mask", "noise"],
        default="both",
        help="Setting augmentation method",
    )
    parser.add_argument(
        "--conduct_pre_train",
        action="store_true",
        help="Whether to conduct pretraining",
    )
    parser.add_argument(
        "--auto_confirm",
        action="store_true",
        help="skip interaction",
    )
    parser.add_argument(
        "--is_projection_head",
        action="store_true",
        help="will user projection head",
    )
    parser.add_argument(
        "--is_shallow",
        action="store_true",
        help="Whether to use shallow model",
    )

    return parser.parse_args()
