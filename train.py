"""Train fresh weights on the shared deduplicated split."""
from experiments import run_training


if __name__ == "__main__":
    run_training("simple_cnn")
