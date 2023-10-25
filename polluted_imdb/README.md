# Situation Description
Sentiment analysis is a very classic classification task in NLP field. <br>
You are provided with a script, train-imdb.py, for training sentiment analysis models based on the Transformer library. <br>
You can first run this script to build a baseline model and check the performance. <br>
If you don't have much computing resource, you can try switch to a smaller pretrained model. <br>
Since the script use a truncated and polluted verion of IMDb dataset, there is a significant drop in evaluation accuracy, plummeting to around 50%.

# Problem Statement
Typically, the accuracies of such models should be over 80%. <br>
So, your goal is to diagnose the reason for the unexpected performance drop and restore the model's accuracy to a more acceptable level. <br>

Please describe your work in the following style, and describe your thinking path as clearly as possible:
1. Analyze the IMDb dataset provided and formulate possible hypotheses.
2. Propose solutions to rectify the issues. If possible, implement your solution to validate its effectiveness.
3. If you find more than one issue, prioritize them based on their potential impact on model performance.
