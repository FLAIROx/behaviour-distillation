# Behaviour Distillation

Code for [Behaviour Distillation](https://openreview.net/forum?id=qup9xD8mW4) (ICLR 2024)

## Distilled Dataset

Distilled datasets are under `results`.

## Running Distillation

Code for running Behaviour Distillation is under `policy_distillation`. Each environment (`brax`, `minatar`, `MNIST`) has its own standalone file.

## Related Work

Behaviour Distillation builds upon other tools in the Jax ecosystem. It notably uses:

- PureJaxRL (https://github.com/luchris429/purejaxrl)
- Gymnax (https://github.com/RobertTLange/gymnax)
- Evosax (https://github.com/RobertTLange/evosax)
- Brax (https://github.com/google/brax)

## Citation
If you use Behaviour Distillation, please cite the following paper:
```
@inproceedings{lupu2024behaviour,
    title={Behaviour Distillation},
    author={Andrei Lupu and Chris Lu and Jarek Luca Liesen and Robert Tjarko Lange and Jakob Nicolaus Foerster},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=qup9xD8mW4}
}
```

Behaviour Distillation is licensed under the Apache 2.0 license.
