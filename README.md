# Behaviour Distillation

Code for Behaviour Distillation (ICML 2024)


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