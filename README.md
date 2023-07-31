# kobert_nsmc_finetuning

### Introduction

From my previous toy project sentiment.kor, I thought it was odd that the model actually converged and performed reasonably well on the test set—I partly knew that I was just lucky. 

This led me to then run the model on another Korean film review dataset “NSMC,” which is one of the benchmarks that is currently used to rate the performance of Korean LLMs. It contains 200k documents while [the previous dataset I used](https://huggingface.co/datasets/sepidmnorozy/Korean_sentiment) contains 40k documents. 

To no surprise, my model was too simple to handle a larger dataset like NSMC and failed to perform well on the test set. While training accuracy went up to 75%, test accuracy stayed at around 50%.

I made some attempts to improve the model. One of them was that I devised a short list of Korean stopwords and excluded them from the positive and negative lexicons. 

```python
with open("stop.txt", "r", encoding="utf-8") as f:
    stop = f.read()
stopwords = stop.splitlines()

# Ttr: Text (training) 
for i in range(len(Ttr)):
    line = Ttr[i]
    for w in line:
        if w not in stopwords:
            if Ytr[i] == 0:
                if w in wc_neg:
                    wc_neg[w] += 1
                else:
                    wc_neg[w] = 1
            else:
                if w in wc_pos:
                    wc_pos[w] += 1
                else:
                    wc_pos[w] = 1
```

This attempt was successful in a way that it boosted up the training accuracy to the low-80s. However, this did not improve the test accuracy at all. In order to make the test accuracy catch up with the training accuracy, I knew it would be easiest to use a completely different approach. 

### Implementation

From a few options that I thought about, I decided to fine-tune a pretrained Korean LLM on the NSMC dataset. 

It did not take long for me to find a Korean BERT model by SKT:

https://github.com/SKTBrain/KoBERT

I was not familiar with the transformers library, so I took this [tutorial](https://huggingface.co/docs/transformers/training) to learn how to load a pretrained model from HuggingFace and train it with the trainer class. 

I was not sure what to do with the hyperparameters, so I left them as default… Haha

https://github.com/sungmogi/kobert_nsmc_finetuning

### Results

<img width="488" alt="Screen Shot 2023-07-28 at 6 15 52 PM" src="https://github.com/sungmogi/kobert_nsmc_finetuning/assets/131221622/6352bb1e-abc4-4ae4-94f0-d108b6636137">

I could see a drastic improvement in accuracy compared to my previous attempt of manually extracting features and putting them through a linear classifier. 

At step 4000, validation loss started to increase while training loss continued to decrease. This clear sign of overfitting can be tackled by stopping early or increasing the dropout rate.
