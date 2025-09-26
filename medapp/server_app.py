"""medapp: A Flower / pytorch_msg_api app."""

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.cli.pull import pull
from flwr.serverapp import Grid, ServerApp
import subprocess
from flwr.serverapp.strategy import FedAvg
from pathlib import Path
from medapp.task import load_centralized_dataset, maybe_init_wandb, test
from medapp.neural_net import Net
import os
# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    num_classes: int = context.run_config["num-classes"]
    lr: float = context.run_config["lr"]
    data_path = context.node_config[context.run_config["dataset"]]

    # Initialize Weights & Biases if set
    use_wandb = context.run_config["use-wandb"]
    wandbtoken = context.run_config.get("wandb-token")
    maybe_init_wandb(use_wandb, wandbtoken)

    # Load global model
    global_model = Net(num_classes=num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(
            num_classes=num_classes,
            use_wandb=use_wandb,
            data_path=data_path,
        ),
    )

    # Save final model to disk
    out_dir = Path(context.node_config["output_dir"])
    print(f"Saving final model to {out_dir}")
    state_dict = result.arrays.to_torch_state_dict()
    print("State dict size:", result.arrays.count_bytes())
    pth = out_dir / "log.txt"
    with open(pth, "w") as f:
        f.write(f"Running with {context.node_config}\n")
        f.write(f"Run ID: {context.run_id}\n")
    print(f"Log file saved to {pth}")
    
    torch.save(state_dict, out_dir / "final_model.pt")
    # read final_model.pt, and print its size
    pth = out_dir / "final_model.pt"
    print("Final model size:", os.path.getsize(pth))

    # print pt size
    # torch.save(state_dict, f"{out_dir}/final_model1.pt")
    # torch.save(state_dict, f"{out_dir}/final_model2.pt")
    # torch.save(state_dict, f"{out_dir}/final_model3.pt")
    # print("Done")


def get_global_evaluate_fn(num_classes: int, use_wandb: bool, data_path: str):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Load model and set device
        model = Net(num_classes=num_classes)
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load entire test set
        test_dataloader = load_centralized_dataset(data_path)

        # Evaluate model on test set
        loss, accuracy = test(model, test_dataloader, device=device)
        metric = {"accuracy": accuracy, "loss": loss}

        if use_wandb:
            wandb.log(metric, step=server_round)

        return MetricRecord(metric)

    return global_evaluate
