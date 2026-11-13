import sys

from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent)
)

import torch

from model import (
    BaselineModel,
    SkipFusionModel,
    count_parameters
)

from config import DEVICE


def main():

    print("Device:", DEVICE)

    # --------------------------------
    # Create test input
    # --------------------------------

    x = torch.randn(
        2,
        3,
        256,
        256
    ).to(DEVICE)


    # --------------------------------
    # Baseline
    # --------------------------------

    baseline = BaselineModel(
        num_classes=2
    ).to(DEVICE)


    baseline_output = baseline(x)


    print()
    print("BASELINE")
    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        baseline_output.shape
    )

    print(
        "Parameters:",
        count_parameters(baseline)
    )


    # --------------------------------
    # Skip fusion
    # --------------------------------

    skip_model = SkipFusionModel(
        num_classes=2
    ).to(DEVICE)


    skip_output = skip_model(x)


    print()
    print("SKIP FUSION")
    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        skip_output.shape
    )

    print(
        "Parameters:",
        count_parameters(skip_model)
    )


    # --------------------------------
    # Verify output size
    # --------------------------------

    assert baseline_output.shape == (
        2,
        2,
        256,
        256
    )

    assert skip_output.shape == (
        2,
        2,
        256,
        256
    )


    print()
    print("MODEL TEST PASSED")


if __name__ == "__main__":

    main()