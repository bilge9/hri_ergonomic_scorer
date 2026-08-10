"""
deba_model_def.py

Single definition of the DEBA network, shared by the training script and the
ROS 2 node.

Previously each file carried its own copy of the architecture and the
checkpoint was a bare state_dict, so a training-side change (a wider layer,
input normalisation) silently stopped matching the node. The node caught the
load error, kept a RANDOMLY INITIALISED network, and published its noise as an
ergonomic score. Two things prevent that here:

  * the architecture lives in one place, and
  * the checkpoint stores its own shape, so the node rebuilds exactly the
    network that was trained instead of guessing.

Input normalisation is held in registered buffers, which means it travels
inside the checkpoint. There is no separate scaler file to forget.
"""

import torch
import torch.nn as nn

# Bumped whenever the checkpoint layout changes in a way the node must notice.
DEBA_CHECKPOINT_VERSION = 2


class DebaMLP(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super(DebaMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden

        # Standardisation constants learned from the training set. Defaults are
        # the identity transform so an un-fitted model still runs.
        self.register_buffer('feat_mean', torch.zeros(input_dim))
        self.register_buffer('feat_std', torch.ones(input_dim))

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def set_normalisation(self, mean, std):
        std = torch.as_tensor(std, dtype=torch.float32).clone()
        std[std < 1e-6] = 1.0   # constant features must not blow up
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(std)

    def forward(self, x):
        return self.network((x - self.feat_mean) / self.feat_std)


def save_checkpoint(model, path, feature_names):
    torch.save({
        'version': DEBA_CHECKPOINT_VERSION,
        'input_dim': model.input_dim,
        'hidden': model.hidden,
        'feature_names': list(feature_names),
        'state_dict': model.state_dict(),
    }, path)


def load_checkpoint(path, map_location='cpu'):
    """
    Rebuild the trained network from a checkpoint.

    Raises on anything unexpected. A DEBA score that silently comes from the
    wrong weights is worse than no DEBA score at all, so the caller is meant
    to disable DEBA on failure rather than carry on.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(blob, dict) or 'state_dict' not in blob:
        raise ValueError(
            "Legacy DEBA checkpoint (bare state_dict) with no architecture or "
            "normalisation metadata. Retrain with train_deba_model.py."
        )
    if blob.get('version') != DEBA_CHECKPOINT_VERSION:
        raise ValueError(
            f"DEBA checkpoint version {blob.get('version')} does not match "
            f"expected {DEBA_CHECKPOINT_VERSION}. Retrain with train_deba_model.py."
        )

    model = DebaMLP(input_dim=blob['input_dim'], hidden=blob['hidden'])
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model, blob.get('feature_names', [])
