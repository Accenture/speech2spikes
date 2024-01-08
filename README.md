README for Speech2Spikes Python Package
===================================

Speech2Spikes is an algorithm designed to convert raw audio data into spike trains that can be utilized by neuromorphic processors using efficient operations (https://doi.org/10.1145/3584954.3584995).

This package is built on top of the PyTorch framework, providing a convenient and efficient solution for transforming audio signals into spiking representations. 

Installation
------------
You can easily install speech2spikes using pip:

```shell
pip install speech2spikes
```

Usage
-----
To convert a batch of audio tensors into spike trains, follow these simple steps:

1. Import the S2S class from the speech2spikes package:
    ```python
    from speech2spikes import S2S
    ```

2. Initialize the S2S object:
    ```python
    s2s = S2S()
    ```

3. Convert the batch of audio tensors to spike trains:
    ```python
    spike_trains = s2s(batch)
    ```
   where `batch` is a list of tuples of form `[(audio, target)]`.

The `spike_trains` output will be a tuple of two tensors, `spikes` and `targets`. Raw audio data is expected to be in the format `(..., timesteps)`.

S2S can also be used as a collate function in PyTorch DataLoaders like this:
```
dl = torch.utils.data.DataLoader(
    train_set,
    batch_size = BATCH_SIZE,
    shuffle = True,
    collate_fn = s2s
)
```

Notes
-----
The Speech2Spikes paper shows spikes as being separated into positive and negative channels however, this code does not separate the spikes this way, instead returning them as signed ints (+1, -1, and 0). We have not seen a decrease in accuracy by doing this and find that reducing the input dimensionality can significantly reduce the overall model size.  
  
Similarly, the cumulative sum feature described in the paper can be replicated using `S2S(cumsum=True)` but is disabled by default due to limited improvements in accuracy while drastically increasing the input dimensionality.

License
-------
Speech2Spikes is made available under a proprietary license that permits using, copying, sharing, and making derivative works from Speech2Spikes and its source code for academics/non-commercial purposes only, as long as the above copyright notice and this permission notice are included in all copies of the software.

See the `LICENSE` file for more information.

Contact
-------
If you have any questions, suggestions, or feedback, please feel free to reach out to us at `neuromorphic_inquiries@accenture.com`.