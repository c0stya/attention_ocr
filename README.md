# Attention-based Optical Character Recognition #

Recurrent neural network with attention mechanism for optical character recognition. A prototype. 

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
python test.py -m model -t "some text"
```
