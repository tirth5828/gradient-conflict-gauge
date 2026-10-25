# How Much of "Gradient Conflict" Is Due to the Tasks, and How Much to the Coordinates?

Tirth Joshi, Katz School of Science and Health, Yeshiva University
DeepMath 2026 (poster)

[Project page](https://tirth5828.github.io/gradient-conflict-gauge/) · [Extended abstract](docs/assets/abstract.pdf) · [Poster](docs/assets/poster.pdf)

## Summary

Rescaling the layers of a bias-free ReLU network, `W_l -> c_l W_l` with step sizes `eta_l -> c_l^2 eta_l` and `prod_l c_l = 1`, leaves the function and the entire gradient-descent trajectory unchanged. The sign of the cosine between two task gradients, which methods such as PCGrad act on, is invariant under this rescaling if and only if the layerwise inner products `a_l = <g_1^l, g_2^l>` share a sign.

- That condition fails in 37.7% of training states.
- Ordinary layerwise conventions flip the sign in only about 4% of states.
- The step size moves the conflict rate from 9% to 83%.

## Code

All models and training loops are written in NumPy (float64). Torch is used only for Hessian-vector products.

```bash
pip install numpy scipy matplotlib torch
```

| Script | Measures | Run |
|---|---|---|
| `conjugation_check.py` | exact conjugacy, sign census, reachability, PCGrad | `python conjugation_check.py` |
| `census_scale.py` | sign census over 1000 seeds | `python census_scale.py run`, then `analyze` |
| `reachability_scale.py` | nearest flipping gauge, layerwise conventions | `python reachability_scale.py run`, then `analyse` |
| `methods_scale.py` | PCGrad, CAGrad and MGDA under the gauge | `python methods_scale.py` |
| `eos_scale.py` | step-size regimes, edge-of-stability projection | `python eos_scale.py` |
| `exploratory.py` | depth, width, biases, Adam, LayerNorm | `python exploratory.py` |
| `mnist_real.py` | MNIST digit and parity tasks, minibatch SGD | `MNIST_DIR=... python mnist_real.py` |

Each script writes the JSON file of the same name in `experiments/` (`conjugation_check.py` writes `results.json`). The per-run `.npz` archives are not included; the scripts regenerate them.

## Citation

```bibtex
@misc{joshi2026gradientconflict,
  title        = {How Much of ``Gradient Conflict'' Is Due to the Tasks,
                  and How Much to the Coordinates?},
  author       = {Joshi, Tirth},
  howpublished = {Poster, DeepMath 2026: Conference on the Mathematical
                  Theory of Deep Neural Networks, Ohio State University},
  note         = {Non-archival extended abstract},
  year         = {2026},
}
```

MIT License.
