# Attention-based Optical Character Recognition #

Recurrent neural network with attention mechanism for optical character recognition. An early prototype. 

[[https://github.com/lightcaster/attention_ocr/blob/master/docs/example.png|alt=attention]]

## Usage ##

To train a new model:
```bash
python train.py 
```
To train an existing model:
```bash
python train.py -m model.pkl
```
To test a model:
```bash
python test.py -m model -t "sometext"
```
