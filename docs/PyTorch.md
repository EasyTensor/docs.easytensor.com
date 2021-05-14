# PyTorch

PyTorch models are packaged in two parts:

1. The serialized parameters (weights and biases) of the model.
2. The model class, including a prediction method.

EasyTensor is able to store and upload the model's parameters (1) automatically, but needs your help in defining the model's architecture and prediction method (2).


Once you have your model class and weights ready, simpl run 

```python
import easytensor
easytensor.pytorch.upload_model("My PyTorch Model", model, "model.py")
```

---

# Saving Model Class

This model class must include a predict_single method that can take in your native input format and return a human readible output. You can think of the predict_single method as where the preprocess, predict, and postprocess happens.

```python
from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        ...

    def predict_single(self, input_instance):
        return input_instance

```

#### predict_single
The `predict_single` method will be called once for every query you run against the inference server. It must accept native input (e.g. text, image bytes, number array) and return a JSON serializable object that will be sent back to the user.

Here are a few input/output examples:

##### Text classification
- `Liverpool beat Manchest United 4-2 on May 13th` -->  `sports`
- `where can i find the best food in town?` --> `location`
- `In the face of overwhelming failure, Tom found success` --> `quotes`


Here is a more complex example that starts up a neural network trained to classify news articles.

```python
from torch import nn
import torch.tensor
from torchtext.datasets import AG_NEWS

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, min_freq=1)

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: int(x) - 1

ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size=95812, embed_dim=64, num_class=4):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
        self.tokenizer = text_pipeline

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, tokens, offset):
        embedded = self.embedding(tokens, offset)
        return self.fc(embedded)

    def predict_single(self, input: str):
        tokenized = torch.tensor(text_pipeline(input))
        output = self.forward(tokenized, torch.tensor([0]))
        prediction = output.argmax(1).item() + 1
        return ag_news_label[prediction]

```