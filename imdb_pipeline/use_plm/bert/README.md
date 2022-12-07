# Bert: Pre-training of deep bidirectional transformers for language understanding 
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- Model instance HuggingFace: [bert-base-uncased](https://huggingface.co/bert-base-uncased)

## Two finetuning strategies: Tradiational finetuning & Prompt-based finetuning
- Tradiational finetuning 
- Prompt-based finetuning

Few shot performance of traditional and prompt finetuning (epoch=5, batch-size=8, n-test=5000):
- 0.5412, 0.7516 (n-train=16)
- 0.8228, 0.812 (n-train=80)
- 0.8818, 0.8654 (n-train=400)

