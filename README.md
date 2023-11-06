# starter code for a3

Add the corresponding (one) line under the `[to fill]` in `def forward()` of the class for ffnn.py and rnn.py

Feel free to modify other part of code, they are just for your reference.

---

Our benmarking performance after completing def forward() code:

**FFNN**

`python ffnn.py --hidden_dim 10 --epochs 1 `
`--train_data ./training.json --val_data ./validation.json`

validation acc >=0.43

**RNN**

`python rnn.py --hidden_dim 32 --epochs 10 `
`--train_data training.json --val_data validation.json`

validation acc >= 0.31

rnn.py is a variation of the RNN algorithm which uses zeros as initial hidden state.
rnn_variation.py is a variation of the RNN algorithm which uses a random initial hidden state.

conda create -n cs6320a2 python=3.8
conda activate cs6320a2
