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

Example output:
```
shap_values2[0,:,1] # 0th sample in the corpus, all players in the game, 1st category in the labels.

.values =
array([ 0.42248802, -0.57413646,  0.57899916, -0.40119484])

.base_values =
0.9585015773773193

.data =
array(['I really wanted to like this movie, because it is refreshingly different from the hordes of everyday horror movie clones, and I appreciate that the filmmakers are trying for something original.',
       "Unfortunately, the plot just didn't hold together and none of the characters were likable enough for me to really care about them or their fates.",
       '<br /><br />Visually, The Toybox was pretty interesting.',
       ' The director took a lot of somewhat risky moves, like adding in little bits of (Flash-looking) animation in parts an'],
      dtype='<U193')
```

