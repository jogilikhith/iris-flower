# Iris species classifier

Python trains a logistic regression model on the classic iris dataset and
exports it to JSON. A plain HTML/CSS/JS page runs it in the browser: move the
sliders and watch the predicted species and probabilities change.

## Run locally
```bash
pip install -r requirements.txt
python train.py                         # optional: docs/model.json is already included
cd docs && python -m http.server 8000   # open http://localhost:8000
```

## Publish on GitHub Pages
Push the repo, then go to **Settings, Pages**, choose **Deploy from a branch**,
Branch `main`, Folder `/docs`. The site appears at
`https://<username>.github.io/<repo-name>/`.

## Structure
```
train.py          trains the model, writes docs/model.json
docs/index.html   page
docs/style.css    styles
docs/script.js    plot, sliders, and the in-browser model
docs/model.json   exported model (generated)
```

## Ideas to extend
- Swap in a decision tree or k-nearest neighbors and compare accuracy
- Try the wine or breast cancer datasets (also built into scikit-learn)
