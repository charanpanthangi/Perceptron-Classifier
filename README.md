# Perceptron Classifier Tutorial & Template

A beginner-friendly, production-ready template for training and evaluating a **single-layer perceptron** classifier on the scikit-learn Iris dataset. The project shows why feature scaling matters, how the perceptron learning rule works, and includes scripts, tests, and a Jupyter notebook.

## What is a Perceptron?
A perceptron is a linear classifier inspired by a biological neuron. It computes a weighted sum of inputs and applies a threshold to decide between classes. The decision boundary is a hyperplane: on one side the model predicts one class, on the other side a different class.

### Learning Rule (Intuition)
For a misclassified example `(x, y)` the perceptron updates its weights with a simple additive rule:

```
w := w + eta * (y - y_hat) * x
```

- `w`: weight vector
- `x`: feature vector
- `y`: true label (+1 / -1 in the binary case)
- `y_hat`: predicted label
- `eta`: learning rate

The update nudges the decision boundary toward correctly classifying the mistake. When data are linearly separable, repeated updates converge to a separating hyperplane.

### When to Use
- Quick linear baseline
- Problems that are roughly linearly separable
- Low-latency, interpretable decision boundaries

### When **Not** to Use
- Non-linearly separable tasks (the perceptron cannot learn curved boundaries)
- Situations needing probability estimates (use logistic regression or other probabilistic models)

## Dataset
The [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) contains 150 flower measurements across three species. It is small, well-balanced, and ideal for demonstrating linear classification.

## Project Structure
```
app/
  __init__.py
  data.py           # load iris dataset
  preprocess.py     # train/test split + scaling (StandardScaler)
  model.py          # perceptron pipeline (scaler + Perceptron)
  evaluate.py       # accuracy, precision, recall, F1, confusion matrix
  visualize.py      # confusion matrix heatmap + PCA scatter plot
  main.py           # end-to-end pipeline
notebooks/
  demo_perceptron_classifier.ipynb  # tutorial notebook
examples/
  README_examples.md
```

## Pipeline Overview
1. **Load** data with `load_iris_dataset()`.
2. **Split & Scale** with `train_test_split_scaled()` using `StandardScaler` (mandatory for stable perceptron updates).
3. **Model** built via `build_perceptron_pipeline()` combining scaling and `sklearn.linear_model.Perceptron`.
4. **Evaluate** using accuracy, precision, recall, F1, and confusion matrix.
5. **Visualize** results with confusion matrix heatmap and PCA scatter plot saved as SVG files.

## Why Scaling Matters
The perceptron update is gradient-like. Large feature magnitudes can dominate the update and slow or prevent convergence. Standardizing features (zero mean, unit variance) balances the influence of each feature and stabilizes learning.

## Getting Started

### Installation
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Pipeline
```
python app/main.py
```
Generated plots are stored in `outputs/`.

### Jupyter Notebook
```
jupyter notebook notebooks/demo_perceptron_classifier.ipynb
```

## Testing
```
pytest
```

## Future Improvements
- Add kernel methods for non-linear boundaries
- Experiment with multi-layer perceptrons (MLPs)
- Try alternative linear solvers such as `SGDClassifier`

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.
