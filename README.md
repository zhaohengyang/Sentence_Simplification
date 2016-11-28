## Sentence rewriting using attentional RNN encoder-decoder model

The project aim to use the most cutting edge deep learning techniques to build a advance model that solve sentence reduction problem. The following are techniques involved in this project.

### Tech 1: Using RNN encoder-decoder sturcture

The idea of Encoder-Decoder architecture is to use use one RNN to read the input sequence to generate a fixed dimensional vector represent the semantic meaning of the sentence which is usually called context or memory. The decoder is another RNN which is trained to use context as input to generate the output sequence.

### Tech 2: Reverse training sequence -- better performance

In this example: 

> Input sequence: russia defense minister ivanov called sunday for the creation of a joint front for combating global terrorism.

> output seqence: russia calls for joint front against terrorism.

Reverse training sequence is a trick that avoid long-term depandencies in RNN. The assumptions is that the input sequence and the output sequence usually has similar word orders.

### Tech 3: Attention mechanism

At each time step, the decoder output is depend on the combination of all the input hidden states instead of just using the last one.

### Visualize system output
![attention visualization png](/img/heatmap.png)

The input sentence are at the bottom. The model prediction is on the left. From the image we can see that great attendent (dark color) is given to relevent words in the input sequence when predicting each word.


### Tech 4: Sampled softmax -- faster in training

Softmax calcualtion is well known the bottle neck in training. Idea of sampled softmax take samples from vocabulary instead of using the full vocabulary. Supose the label at time t is "after", we select 200 word from batched input and use it as nagative samples. The predicted y will be a probability distribution over 201 word (200 select word plus true y) instead of the probability distribution over fll vocabulary like 40,000 words. The time cost on this calcualtion bottom neck will reduced by 20 times.

### Tech 5: Bucket -- faster in training

During training, we use batch to keep the speed. Since input sequence has different length, in order to use batch, we need to pad sequences into same length. However padding is envolved in matrix computation but not helping for model's performance. The idea of bucket is to minimize padding thus save some computational power.
Assume we have 100 inputs that around 10 words and another 100 inputs that around 20 words, instead of pad all the sentence to 20 word length, we could put them into two buckets, one with 10 word length and the other with 20 word length.

### Tech 6: Dealing with out of vocabulary problem

#### Problem distription:
Out of vocabulary is a common problem in sequence to sequence model. Usually we use one token to represent all the unknown words and this token is **\<unk\>**. 

When the **\<unk\>** token shows up in input sequence, it nagatively impact on the model to interpret the meaning of the input. Since we use same token **\<unk\>** to represent two different words, in this case russia and terrorism, the two word become interchangeable from the model understanding.

On the other hand, when we encounter **\<unk\>** in output sequence during training, the prediction trend to stuck at **\<unk\>**. Like the following.

> Model input:

```
Input sequence: russia defense minister ivanov called sunday for the creation of a joint front for combating global terrorism.

output sequence: russia calls for joint front against terrorism.
```

> After preprocessing:

```
Input sequence: <unk> defense <unk> <unk> called sunday for the creation of a joint front for combating global <unk>. 

output sequence: <unk> calls for joint front against <unk>.
```

> Model prediction:

```
prediction: calls <unk> <unk> <unk> <unk> <unk> <unk> <unk>.
```

#### Solution: \<unk\> placeholders

We use 100 unk placeholders to represent out of vocabulary words, embedings for these unknown words are less interchangeable in encoder. The main advantage is prediction has much lower chance to stuck with **\<unk\>** since the probablity of seen **\<unk\>** during trianing is evenly split out into 100 placeholders. 

### Tech 7: Copy mechanism. 

The idea is to train the model to learn to copying keywords from the input sequence instead of generating output based on their meanings. The well trained model tend to make a copy when the word itself is unknown but the context tell the model to use the word. We followed neuralmonkey's work (https://github.com/ufal/neuralmonkey) and implement a LSTM decoder with copy mechanism. This mechanism help our model dealing with out of vocabulary problem, since the copy mechanism is based on word location in the sentence instead of word meaning, unknown word can be copy to the right location without knowning the word.

### Tech 8: Training smoothly

#### Problem: gap between training and inference

During decoding phase in training, we fed true previous target tokens as input at each time step. During inference, however, true previous target tokens are unavailable. We thus use the previous tokens generated by the model itself, yielding a gap between how the model is used at training and inference.

The main problem is that mistakes made early in the sequence generation process are fed as input to the model and can be quickly amplified because the model might be in a part of the state space it has never seen at training time.

#### Solution: Scheduled Sampling

The idea is to mix to situation. The simple solution is to take 95% time for feeding the true previous target as current input and 5% time for feeding model generated previous output as current input. Instead hard coded designed as 95%, we followed Samy Bengio's work that dynamically adjust the balance between target feeding and self generation. As the model training, the self generation rate tend to increase as narrowing the gap between training and inference.
