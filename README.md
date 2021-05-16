# cityscapes_segmentation_tf2

*In progress*

An end-to-end semantic segmentation pipeline on the cityscapes dataset, using HRNetV2

Current experimentaion: how can we replicate SOTA results when using a single consumer GPU, rather than 8xV100s?
- Use mixed precision to cut down on VRAM usage
- In order to use larger batch sizes, we aggregate the graidents over multiple batches. This is done using a custom training loop defined in `train_utils.py`. Now we can use a batch size of 2, but with an "effective" batch size of 6 or 12. 
- But we still have to problem of batch statistics being computed over only a couple examples. To solve this, group normalization is used instead. 

```python

```
