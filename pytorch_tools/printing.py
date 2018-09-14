"""
docstring
"""

def print_model_stats(index, batch_size, len_data, loss):
    """
    Print running model statistics.

    Parameters:
    -----------
    index: int
        batch index
    batch_size: int
        size of the batch
    len_data: int
        number of examples in the dataset
    loss: float
        current batch loss

    Returns:
    --------
    nothing, prints to screen
    """
    x = "    {n}/{d}: loss = {loss:.4f}          "
    print(x.format(n=index*batch_size, d=len_data, loss=loss),
          end="\r", flush=True)
