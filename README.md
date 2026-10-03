# neural-network-from-scratch-
A feed-forward neural network built from raw PyTorch tensor operations. The forward pass, loss, backpropagation and optimiser are all hand-written, with no `torch.nn`, no autograd and no `torch.optim`. It reaches **98.35% ± 0.08% test accuracy on MNIST** (5 seeds), matching scikit-learn's `MLPClassifier` (98.28%) trained with the same recipe.

[**Run the notebook in Colab**](https://colab.research.google.com/drive/1Sxr-4DuxvT1TMQ5G16yWzT2QnOUn1TLq) · [`mlp_from_scratch.ipynb`](mlp_from_scratch.ipynb)

## Why this project

Frameworks hide the parts of a neural network that matter most when something goes wrong: the gradient flow, the numerical edge cases and the gap between the loss you optimise and the metric you care about. I built every piece myself, verified each one independently, and then evaluated the result the way I would a production model: proper data splits, baselines, multiple seeds and an error analysis.

## What's inside

| Stage | What it does |
|---|---|
| **Data** | Two moons (2D, for visualising decision boundaries) and MNIST (784-D, 10 classes, the main problem). Scaled, flattened and split with a fixed seed: 50k train / 10k validation / 10k test. |
| **Model** | One-hidden-layer MLP, `logits = ReLU(XW1 + b1)W2 + b2`, with He initialisation. The forward pass returns logits plus a cache for backprop. |
| **Loss** | Numerically stable log-softmax and mean cross-entropy, computed from logits. |
| **Backprop** | Gradients of all four parameter tensors derived by hand and written as explicit matrix operations. |
| **Training** | Mini-batch SGD, per-epoch train and validation metrics, early stopping that restores the best weights. |
| **Tuning** | Two-stage grid search over learning rate, width and batch size, scored on validation data only. |
| **Evaluation** | The chosen configuration retrained on 5 seeds, the test set used once, then baselines, error analysis and loss versus accuracy. |

The code uses the maths's own names (`W1, b1, Z1, A1, dZ2, dW1`) and every forward and backward line carries a shape comment, e.g. `# (B, 768)`.

## Results

| Model | MNIST test accuracy |
|---|---|
| Majority class | 11.35% |
| Untrained network (100 random initialisations) | 10.4% ± 2.7% |
| Nearest-mean digit template (hand-written rule) | 81.99% |
| **This MLP, 784 → 768 → 10 (610,570 parameters), 5 seeds** | **98.35% ± 0.08%** |
| scikit-learn `MLPClassifier`, same recipe (reference only) | 98.28% |

- Tuning cut validation errors from 2.26% to 1.73%. Validation (98.27%) and test (98.35%) agree within one standard deviation, so selecting among 66 configurations did not overfit the validation set.
- On two moons the network reaches 97.7% ± 0.4%, and decision boundaries are plotted for hidden widths 2, 4, 16 and 64 to show underfitting and capacity directly.

## Engineering details

**Numerical stability.** The textbook softmax overflows on logits like `[1000, 999]` and underflows on `[0, -200]`. I subtract the per-row maximum and compute log-probabilities directly. The maximum has to be taken per row: a single batch-wide maximum silently underflows the other rows.

**Verifying backprop without autograd.** Each component has its own check:

- Forward pass: matched against a network computed by hand.
- Loss: hand-worked values, plus the overflow and underflow cases above.
- Backward pass: a float64 central-difference gradient check over all 82 parameters of the moons network. The largest difference was 2.0 × 10⁻¹¹ against a 10⁻⁷ threshold.
- Training loop: it memorises a 64-image batch to 100% accuracy.

**Initialisation.** He initialisation compensates for ReLU zeroing about half its inputs. The drawn weight standard deviation (0.0506) matches its target (0.0505).

**Overfitting and early stopping.** On MNIST, train loss kept falling while validation loss stayed flat. Early stopping on validation loss restores the best weights. I stop on loss rather than accuracy because accuracy moves in coarse steps and plateaus while loss still improves.

**Fair hyperparameter selection.** The selection rule was fixed before running anything, all configurations shared seeds, and the grid was extended when the first winner sat on its edge. A "winner's curse" showed up too: batch size 64 led on one seed and lost over three, so the deciding grid used five seeds per configuration.

## Analysis

- **Where it fails.** The largest errors are pairs of digits that share strokes: 4→9 (51 errors summed over seeds), 7→2 (33), 5→3 (30). They are one-directional: 4→9 happens 51 times but 9→4 only 20.
- **Loss versus accuracy.** The 1.68% of test images the model gets wrong carry 84% of the test loss, and about a third of them are predicted with over 90% confidence. Cross-entropy rewards confidence, so the best-accuracy configuration was not the lowest-loss one. The report works through this and how it would be monitored in deployment.
- **No spatial structure.** An MLP treats pixels as an unordered vector, so it must learn each stroke at every position separately. This explains the stroke-sharing confusions and motivates the next step.

## Limitations and next steps

Depth, momentum/Adam and weight decay were not explored, so conclusions hold for one hidden layer and plain SGD. The model has no "not a digit" output, and it is not calibrated. The natural extension is a convolutional network trained under the identical protocol and compared on shifted images, with a shift-and-rotate-augmented MLP as the control.

## Running it

The notebook runs on CPU or GPU (`cuda` is used when available) and on Colab as-is: PyTorch, torchvision, scikit-learn and matplotlib come preinstalled.

```bash
pip install torch torchvision scikit-learn matplotlib numpy
jupyter notebook mlp_from_scratch.ipynb
```

The hyperparameter sweep is behind a `RUN_SWEEP` flag. It takes about 40 minutes on an RTX 4080 SUPER and several times longer on a Colab T4. Set it to `False` to reuse the recorded winners. Everything is seeded for reproducibility.

## What counts as "from scratch"

Only tensor creation, indexing, `@`, element-wise maths (`exp`, `log`, `clamp`), reductions and seeding are used. No `torch.nn`, autograd, `torch.optim` or prebuilt softmax and cross-entropy helpers. `make_moons` and the raw MNIST tensors are used for data only. scikit-learn's `MLPClassifier` appears once, as an evaluation reference, with its accuracy computed by my own code.

## Tech

Python · PyTorch (tensor ops only) · NumPy · Matplotlib · scikit-learn (data and one reference model)
