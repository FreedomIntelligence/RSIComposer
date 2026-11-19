"""Training batches independent of benchmark semantics."""


def training_groups(dataset, batch_size, accumulation, world):
    """Global groups without sampler padding or repeated training examples."""
    width = batch_size * accumulation * world
    return [dataset[i:i + width] for i in range(0, len(dataset), width)]
