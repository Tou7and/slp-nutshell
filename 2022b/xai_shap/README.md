# Template
Use SHAP to analyse the contribution of text components(words, sentences, ...) of inference results. <br>
- reference: [SHAP-text-examples](https://shap.readthedocs.io/en/latest/text_examples.html#sentiment-analysis)

Load libraries and dataset: 
```
import transformers
import datasets
import shap
import numpy as np

dataset = datasets.load_dataset("imdb", split="test")

# shorten the strings to for the transformer, and select only a few
short_data = [v[:512] for v in dataset["text"][12627:12647]]
```

Load pre-trained classifier, and construct SHAP explainer:
```
classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)

# This is the default explainer which treat words as default players
explainer = shap.Explainer(classifier)

# Pass a regular expression to create the text masker, and build a explainer that treat sentences as players
masker = shap.maskers.Text(r"\.")
explainer2 = shap.Explainer(classifier, masker)

```

Get SHAP values and make beautiful plots:
```
shap_values2 = explainer2(short_data[-8:]) # select samples to get SHAP

# Run this line in Jupyter Lab to view the SHAP plots
shap.plots.text(shap_values3[:,:,1]) # select label to view
``` 

