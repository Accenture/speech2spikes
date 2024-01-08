import torch
import torchaudio
import matplotlib.pyplot as plt

from pathlib import Path
from speech2spikes import S2S

def test_s2s():
    sample_file = Path(__file__).parent.joinpath("0a2b400e_nohash_0.wav")
    raw_audio, sample_rate = torchaudio.load(sample_file)

    batch = []
    for i in range(500):
        batch.append(raw_audio)
    labels = [torch.Tensor(0)] * 500

    s2s = S2S()
    tensors, targets = s2s(list(zip(batch, labels)))

    assert len(tensors) == 500
    assert len(targets) == 500
    assert tensors.shape == torch.Size([500, 1, 20, 201])
    assert tensors.sum() != 0

    s2s = S2S(cumsum=True)
    tensors, targets = s2s(list(zip(batch, labels)))

    assert len(tensors) == 500
    assert len(targets) == 500
    assert tensors.shape == torch.Size([500, 1, 40, 201])
    assert tensors.sum() != 0

if __name__ == "__main__":
    test_s2s()