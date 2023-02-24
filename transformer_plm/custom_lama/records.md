# Dialogue Records 
TODO: 
- keep good questions and remove bad ones
- Considr use Google Search answers as Ground-truth, calculate ROGUE scores for benchmarks

## "Who is the greatest emperor in the history, Shi Huangdi or Tiberius? Give the rationale before answering."
- FLAN-T5-XL: ['Tiberius was the Roman Emperor from AD 79 to AD 89. Shi Huangdi was the Emperor of the Qin Dynasty from AD 279 to AD 286. So the final answer is Tiberius.']
- FLAN-T5-small: ['Shi Huangdi is the greatest emperor in the history. Tiberius is the emperor of the Xi Jinping Dynasty. The final answer: Tiberius.']
  
## "Who is the greatest emperor in the history? Give the rationale before answering."
- FLAN-T5-XL: ['Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the empires. Emperors rule over the']
- FLAN-T5-small: ['Emperor Constantinople is the greatest emperor in the history. The final answer: Constantinople.']

## "What is the biggest threat to America, and what kind of damage can it make?"
- FLAN-T5-XL: ['a nuclear attack can cause tens of thousands of deaths and billions of dollars in damage.']
- FLAN-T5-small: ['a hurricane']

## "What is the differnece between CPU and GPU? Give the rationale before answering."
- FLAN-T5-XL: ['The difference between CPU and GPU is the amount of processing power. The difference between CPU and GPU is the amount of memory. The difference between CPU and GPU is the amount of memory. So the final answer is memory.']
- FLAN-T5-small: ['CPUs are a type of processor that is used for a number of applications. GPUs are a type of processor that is used for a number of applications. So the final answer is processor.']

## "Which one is more delicious, beef or chicken? Give the rationale before answering."
- FLAN-T5-XL: ['Beef is a protein that is high in iron and zinc. Chicken is a protein that is low in iron and zinc. So the final answer is chicken.']
- FLAN-T5-small: ['Chicken is a dish that is cooked with a lot of meat. Beef is a dish that is cooked with a lot of meat. The final answer: beef.']

## "Which one is more nutritious, beef or chicken? Give the rationale before answering."
- FLAN-T5-XL: ['Beef is a source of protein and iron. Chicken is a source of protein and iron. So the final answer is chicken.']
- FLAN-T5-small: ['Chicken is a type of food. Chicken is a type of food. The final answer: chicken.']

## Trolley problem
```
text= """
There is a runaway trolley barreling down the railway tracks. Ahead, on the tracks, there are five people tied up and unable to move. The trolley is headed straight for them. You are standing some distance off in the train yard, next to a lever. If you pull this lever, the trolley will switch to a different set of tracks. However, you notice that there is one person on the side track. You have two (and only two) options:
Do nothing, in which case the trolley will kill the five people on the main track.
Pull the lever, diverting the trolley onto the side track where it will kill one person.
Which is the more ethical option? Or, more simply: What is the right thing to do?
"""
```
- FLAN-T5-XL: ['Pull the lever, diverting the trolley onto the side track where it will kill one person.']
- FLAN-T5-small: ['Do nothing']

