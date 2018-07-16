def set_spawn_enabled():
    """Set PyTorch start method to spawn a new process rather than spinning up a new thread.

    This change was necessary to allow multiple DataLoader workers to read from an HDF5 file.

    See Also:
        https://github.com/pytorch/pytorch/issues/3492
    """
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
