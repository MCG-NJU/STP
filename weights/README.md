# Weights

Place the released pre-training checkpoint here:

    0.95_aug0.8.pth

It is a 1.75 GB torch (>=1.13, zip format) checkpoint holding
`{model, optimizer, epoch, scaler, args}`.

Verify it against the code before using it:

    python tools/verify_ckpt_params.py weights/0.95_aug0.8.pth

This directory is git-ignored (see `.gitignore`) — publish the checkpoint via a
release asset or a model hub rather than committing it.
