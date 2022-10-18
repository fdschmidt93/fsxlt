# Examples

This document comprises a few tips and tricks to illustrate common concepts with `trident`.


# How to subsample a dataset?

```yaml
dataset_cfg:
  _method_:
    shuffle:
      seed: 42
    select:
      indices:
        _target_: builtins.range
        _args_:
          - 0
          - 1000
````

# How to restore weights for a declared model?
```
module:
  model:
    pretrained_model_name_or_path: ???
    num_labels: ???
  module_cfg:
    weights_from_checkpoint:
      ckpt_path: "FULL_PATH_TO_WEIGHTS"
```
