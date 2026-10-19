"""Train an iris species classifier and export it to docs/model.json for the web page."""
import json
import math
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# 1. Measure accuracy on flowers the model has not seen
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
accuracy = LogisticRegression(C=10, max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te)
print(f"Test accuracy: {accuracy:.1%}")

# 2. Refit on all 150 flowers and export
model = LogisticRegression(C=10, max_iter=1000).fit(X, y)
payload = {
    "classes": list(iris.target_names),
    "coef": model.coef_.tolist(),
    "intercept": model.intercept_.tolist(),
    "range": [[math.floor(X[c].min() * 10) / 10, math.ceil(X[c].max() * 10) / 10] for c in X.columns],
    "mean": X.mean().tolist(),
    "points": [[float(a), float(b), int(c)] for a, b, c in zip(X.iloc[:, 2], X.iloc[:, 3], y)],
    "accuracy": accuracy,
}
out = Path("docs/model.json")
out.write_text(json.dumps(payload))
print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB)")
