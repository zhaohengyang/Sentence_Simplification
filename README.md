{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Capstone: sentence rewriting using attentional RNN encoder-decoder model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Overall goal\n",
    "The goal of this project is simple: As the user writing the sentence, the system output a summarization of the sentence. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Brief introduction of RNN encoder-decoder"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The idea of Encoder-Decoder architecture is to use use one RNN to read the input sequence to generate a fixed dimensional vector represent the semantic meaning of the sentence which is usually called context or memory. The decoder is another RNN which is trained to use context as input to generate the output sequence. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img  src=\"seq2seq.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Reverse training sequence: better performance"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Input sequence: russia defense minister ivanov called sunday for the creation of a joint front for combating global terrorism.\n",
    "\n",
    "output seqence: russia calls for joint front against terrorism."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Reverse training sequence is a trick that avoid long-term depandencies in RNN. It works when the input sequence and output sequence has similar word orders."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Attention mechanism\n",
    "At each time step, the decoder output is depend on the combination of all the input hidden states instead of just using the last one. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img  src=\"attention.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualation of attention mechanism"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here is an Attention visualization example from Rush et al."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img  src=\"attention_visualization.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sampled softmax: faster in training"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Where is training bottle neck?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img  src=\"rnn.jpg\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$\n",
    "\\begin{aligned}\n",
    "x_t & \\in \\mathbb{R}^{300} \\\\\n",
    "o_t & \\in \\mathbb{R}^{40,000} \\\\\n",
    "s_t & \\in \\mathbb{R}^{512} \\\\\n",
    "U & \\in \\mathbb{R}^{300 \\times 512} \\\\\n",
    "V & \\in \\mathbb{R}^{512 \\times 40,00} \\\\\n",
    "W & \\in \\mathbb{R}^{512 \\times 512} \\\\\n",
    "y_t & \\in \\mathbb{R}^{40,000}\\\\\n",
    "\\end{aligned}\n",
    "$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$\n",
    "\\begin{aligned}\n",
    "s_t &= \\tanh(Ux_t + Ws_{t-1}) \\\\\n",
    "o_t &= \\mathrm{softmax}(Vs_t) \\\\\n",
    "loss &= y_to_t\n",
    "\\end{aligned}\n",
    "$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$\n",
    "\\begin{aligned}\n",
    "O(s_t) &= 300 \\times 512 + 512 \\times 512 &= 812 \\times 512\\\\\n",
    "O(o_t) &&= 40,000 \\times 512 \\\\\n",
    "loss &&= 40,000\n",
    "\\end{aligned}\n",
    "$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Softmax calcualtion is the bottle neck in training. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Idea of sampled softmax take samples from vocabulary instead of using the full vocabulary. Supose the label at time t is cook, we select 200 word from batched input and use it as nagative samples. The $y_{new}$ label is a one-hot vector only contain cook itself and nagative samples, cook will be valued as 1 and nagative samples will be valued as 0. $V_{new}$ is slicing corresponding sample from $V$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$\n",
    "V_{new} \\in \\mathbb{R}^{512 \\times 200} \\\\\n",
    "y_{new}  \\in \\mathbb{R}^{200}\\\\\n",
    "$\n",
    "\n",
    "$\n",
    "\\begin{aligned}\n",
    "newO(o_t) &= 200 \\times 512 \\\\\n",
    "newloss &= 200\n",
    "\\end{aligned}\n",
    "$\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bucket: faster in training"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "During training, we use batch to keep the speed. Since input sequence has different length, in order to use batch, we need to pad sequences into same length. However padding is envolved in matrix computation but not helping for model's performance. The idea of bucket is to minimize padding thus save some computational power.\n",
    "Assume we have 100 inputs that around 10 words and another 100 inputs that around 20 words, instead of pad all the sentence to 20 word length, we could put them into two buckets, one with 10 word length and the other with 20 word length."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Unk problem: out of vocabulary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Out of vocabulary is a common problem in sequence to sequence model. Usually we use one token to represent all the unknown words and this token is <font color='blue'>$<unk>$</font>. \n",
    "\n",
    "When the <font color='blue'>$<unk>$</font> token shows up in input sequence, it nagatively impact on the model to interpret the meaning of the input. Since we use same token <font color='blue'>$<unk>$</font> to represent two different words, in this case russia and terrorism, the two word become interchangeable from the model understanding.\n",
    "\n",
    "On the other hand, when we encounter <font color='blue'>$<unk>$</font> in output sequence during training, the prediction trend to stuck at <font color='blue'>$<unk>$</font>. Like the following."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Input sequence: <font color='blue'>russia defense minister ivanov called sunday for the creation of a joint front for combating global terrorism.</font>\n",
    "<br />\n",
    "output sequence: <font color='green'>russia calls for joint front against terrorism.</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Input sequence: <font color='blue'>$<unk>$ defense minister ivanov called sunday for the creation of a joint front for combating global $<unk>$. </font>\n",
    "\n",
    "output sequence: <font color='green'>$<unk>$ calls for joint front against $<unk>$.</font>\n",
    "\n",
    "prediction: <font color='red'>$<unk>$ $<unk>$ $<unk>$ $<unk>$ $<unk>$ $<unk>$ $<unk>$.</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Our solution: unk placeholders"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We use 100 unk placeholders to represent out of vocabulary words, embedings for these unknown words are less interchangeable in encoder. The main advantage is prediction has much lower chance to stuck with <font color='blue'>$<unk>$</font> since there are so many different <font color='blue'>$<unk>$</font>, it don't know which one to stuck on~"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Input sequence: <font color='blue'>$<unk1>$ defense minister ivanov called sunday for the creation of a joint front for combating global $<unk2>$. </font>\n",
    "\n",
    "output sequence: <font color='green'>$<unk1>$ calls for joint front against $<unk2>$.</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Other solutions:\n",
    "1. Char-based model don't have unk problem since all word are represent as a sequence of character. Major draw back is input sequence is too long to represent a normal sentences, for a sentence with 10 words, the sequecne of character could be around 70. Facing servious long-term dependency problems.\n",
    "2. Mapping unknown words back to known word by the similarity of word embeding. It won't solve the unk problem but slightly helpful to the model. We'd been adopted this trick.\n",
    "3. Use upper mentioned sampled softmax or hierarchical softmax to increase the training speed, so we can put more vocabulary into the model. \n",
    "4. Copy mechanism. The idea is to train the model to learn to copying keywords from the input sequence instead of generating output based on their meanings. The well trained model tend to make a copy when the word itself is unknown but the context tell the model to use the word. We plan to implement this technique in our future work."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Problem: gap between training and inference"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "During decoding phase in training, we fed true previous target tokens as input at each time step.\n",
    "\n",
    "During inference, however, true previous target tokens are unavailable. We thus use the previous tokens generated by the model itself, yielding a gap between how the model is used at training and inference.\n",
    "\n",
    "The main problem is that mistakes made early in the sequence generation process are fed as input to the model and can be quickly amplified because the model might be in a part of the state space it has never seen at training time."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img  src=\"seq2seq_training_new.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img  src=\"test_generation.jpg\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Our solution: fusion between label feedding and self-generate\n",
    "We take 95% time for feeding the true previous target as current input and 5% time for feeding model generated previous output as current input.\n",
    "### Other solution:\n",
    "Scheduled Sampling: Instead hard coded designed as 95%, Samy Bengio introduce an algorithm that dynamically adjust the balance between target feeding and self generation. As the model training, the self generation rate tend to increase as narrowing the gap between training and inference."
   ]
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
