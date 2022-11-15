""" Prompt learning using BERT on IMDb dataset.

Reference: https://github.com/thunlp/OpenPrompt/blob/main/tutorial/0_basic.py

Results: TBD

2022.11.14, JamesH.
"""
import os
import torch
from datasets import load_dataset
from transformers import AdamW # get_linear_schedule_with_warmup
# from torch.optim import AdamW
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification

def main(use_cuda=True, data_folder="exp"):
    """ prompt learning on BERT for sentiment classification """
    print(f"----------- Raw dataset from: {data_folder} -------------")
    datafiles = {
        'train': os.path.join(data_folder, 'train.json'), 
        'test': os.path.join(data_folder, 'test.json')}
    raw_dataset = load_dataset("json", data_files=datafiles, cache_dir="exp/cache", field="data")

    # Load PLM
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-uncased")

    # Prompt(template) engineering
    dataset = {}
    for split in ['train', 'test']:
        dataset[split] = []
        for idx, data in enumerate(raw_dataset[split]):
            input_example = InputExample(
                text_a=data['text'],
                label=int(data['label']),
                guid=idx)
            dataset[split].append(input_example)

    prompt_template = ManualTemplate(
        text='{"placeholder":"text_a"} It was {"mask"}',
        tokenizer=tokenizer,)

    # Answer engineering
    prompt_verbalizer = ManualVerbalizer(
        num_classes=2,
        label_words=[["bad"], ["good"]], 
        tokenizer = tokenizer)

    wrapped_tokenizer = WrapperClass(
        max_seq_length=512,
        decoder_max_length=3,
        tokenizer=tokenizer,
        truncate_method="head")

    # Prepare input training & testing data for the PLM
    model_inputs = {}
    for split in ['train', 'test']:
        model_inputs[split] = []
        for sample in dataset[split]:
            tokenized_example = wrapped_tokenizer.tokenize_one_example(
                prompt_template.wrap_one_example(sample),
                teacher_forcing=False)
            model_inputs[split].append(tokenized_example)

    # Prepare Pytorch DataLoader
    train_dataloader = PromptDataLoader(
        dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
        batch_size=8, shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    # Start normal PLM training
    print("Now, start normal PLM training...")
    prompt_model = PromptForClassification(
        plm=plm, template=prompt_template,
        verbalizer=prompt_verbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model = prompt_model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5) # 5e-5, 1e-4

    for epoch in range(5):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step %100 ==1:
                # print(logits)
                # print("-"*50)
                # print(labels)
                print(f"Epoch {epoch}, Step {step}, average loss: {tot_loss/(step+1)}", flush=True)

    print("Start testing model performance...")
    prompt_model.eval()
    test_dataloader = PromptDataLoader(
        dataset=dataset["test"], template=prompt_template, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
        batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")
    
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    # print(alllabels[:10])
    # print(allpreds[:10])
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print("ACC:", acc)
    return acc

if __name__ == "__main__":
    main(data_folder="../exp/few_n0016")
    main(data_folder="../exp/few_n0080")
    main(data_folder="../exp/normal_n0400")
