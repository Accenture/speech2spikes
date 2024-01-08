import torch
import torchaudio


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def tensor_to_events(batch, threshold=1, device=None):
    """Converts a batch of continuous signals to binary spikes via delta
    modulation (https://en.wikipedia.org/wiki/Delta_modulation).

    Args:
        batch: PyTorch tensor of shape (..., timesteps)
        threshold: The difference between the residual and signal that
            will be considered an increase or decrease. Defaults to 1.
        device: A torch.Device used by PyTorch for the computation. Defaults to 
            None.

    Returns:
        A PyTorch int8 tensor of events of shape (..., timesteps).

    TODO:
        Add support for using multiple channels for polarity instead of signs
    """
    events = torch.zeros(batch.shape)
    levels = torch.round(batch[..., 0])
    if device:
        events = events.to(device)

    for t in range(batch.shape[-1]):
        events[..., t] = (batch[..., t] - levels > threshold).to(torch.int8) - (
            batch[..., t] - levels < -threshold
        ).to(torch.int8)
        levels += events[..., t] * threshold
    return events


class S2S:
    """The S2S class manages the conversion from raw audio into spikes and
    stores the required conversion parameters.

    Attributes:
        device: A torch.Device used by PyTorch for the computation. Defaults to 
            None.
        labels: A list of labels. The index of the label will be used as the
            target. Defaults to None
    """

    def __init__(self, cumsum=False, device=None, labels=None):
        self.cumsum = cumsum
        self.device = device

        self.labels = labels
        self._default_spec_kwargs = {
            "sample_rate": 16000,
            "n_mels": 20,
            "n_fft": 512,
            "f_min": 20,
            "f_max": 4000,
            "hop_length": 80,
        }
        self.transform = torchaudio.transforms.MelSpectrogram(
            **self._default_spec_kwargs
        )

    def __call__(self, batch):
        """Simple wrapper of convert for completeness"""
        return self.convert(batch)

    def configure(self, labels=None, threshold=1, **spec_kwargs):
        """Allows the user to configure parameters of the S2S class and the
        MelSpectrogram transform from torchaudio.

        Go to (https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html)
        for more information on the available transform parameters.

        Args:
            labels: A list of labels. The index of the label will be used as the
                target. Defaults to None
            threshold: The difference between the residual and signal that
                will be considered an increase or decrease. Defaults to 1.
            **spec_kwargs: Keyword arguments pass to torchaudio's MelSpectrogram
        """
        self.labels = labels
        self.threshold = threshold

        spec_kwargs = {**self._default_spec_kwargs, **spec_kwargs}
        self.transform = torchaudio.transforms.MelSpectrogram(spec_kwargs)

    def convert(self, batch):
        """Converts raw audio data to spikes using Speech2Spikes algorithm
        (https://doi.org/10.1145/3584954.3584995)

        Args:
            batch: List of tensors and corresponding targets [(tensor, target)]

        Returns:
            (tensors, targets):
                tensors: PyTorch int8 tensor of shape (batch, ..., timesteps)
                targets: A tensor of corresponding targets. If labels are
                    provided, this will convert labels to indices.

        TODO:
            Add support for single sample conversion
        """
        tensors, targets = [], []
        for waveform, label in batch:
            tensors += [waveform]
            if self.labels:
                targets += [torch.tensor(self.labels.index(label))]
            else:
                targets += [label]

        tensors = pad_sequence(tensors)
        tensors = self.transform(tensors)
        if self.cumsum:
            csum = torch.cumsum(tensors, dim=-1)
            # Concatenate csum and tensors on mel channel dimension
            tensors = torch.cat((csum, tensors), dim=2)
        tensors = torch.log(tensors)
        tensors = tensor_to_events(tensors, device=self.device)

        targets = torch.stack(targets)

        return tensors, targets
