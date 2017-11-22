# seq2seq-pytorch
Pytorch implementation of Seq2Seq model for conversational agents

## Requirements
CPU or CUDA version 8.0

## Get started
Unzip `data/cornell_movie_dialogs_corpus.zip`.
Run `python src/data.py` for processing the data.
Run `python src/runner.py` for training.

## TODO
* ~~Print conversation examples every epoch (make sure it's working)~~
* ~~Print BLEU scores every epoch~~
* ~~Change SGD to Adam~~
* ~~Change padding to masking~~
* Add memory pad
* Use settings from [this repo](https://github.com/jiweil/Neural-Dialogue-Generation)
* Handle/Remove very short sequences
* Add TensorBoard
* Add attention mechanism
