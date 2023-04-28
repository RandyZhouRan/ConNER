# ConNER: Consistency Training <br/> for Cross-lingual Named Entity Recognition
This repository contains the code for EMNLP 22 paper "ConNER: Consistency Training for Cross-lingual Named Entity Recognition"
![alt text](https://github.com/RandyZhouRan/ConNER/blob/main/ConNER.png?raw=true)

## Requirements
* Python >= 3.7
* PyTorch >= 1.7
* Transformers >= 3.5
* conlleval = 0.2

Our experiments are run on a single Nvidia V100 32GB GPU.

## Data Format

We have released the preprocessed data used in our experiments under the `data` directory. To train on your customized datasets, follow the data format as below.

Training `train.txt`, validation `dev.txt` and testing `test.txt` files adopt the CoNLL two-column format. Unlabeled data file `unlabel.txt` and its translation `trans.txt` use an extra column in between to indicate span-alignment, which can be automatically obtained using our [alignment-free translation](#alignment-free-translation). 

## Training

To train ConNER on cross-lingual NER (for example, English to German transfer), use the following commands:
```
cd model/En2De
bash 01_xlmr_train_eval.sh
```
All hyperparameters, such as loss weight coefficients `UDA_W`, `RDROP_W`, batch size `BS` are specified in the bash script.

## Alignment-free Translation
Place your Google API key `google_translate.json` under the `translate` directory.

Run the following commands to conduct alignment-free translation on `translate/sample.txt`

```commandline
cd translate
python 01_translate.py  # translate unlabeled data
python 02_clean.py      # cleaning the raw translated data
python 03_lin2cols.py   # convert inline sentences into CoNLL's two column format
python 04_col2align.py  # add span alignment indicator
```

The generated files `unlabel.txt` and `trans.txt` will be used for ConNER training.

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{zhou2022conner,
  title={ConNER: Consistency Training for Cross-lingual Named Entity Recognition},
  author={Zhou, Ran and Li, Xin and Bing, Lidong and Cambria, Erik and Si, Luo and Miao, Chunyan},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year={2022}
}
```

## Acknowledgements
The training code in this repo has been built upon [pytorch-neural-crf](https://github.com/allanj/pytorch_neural_crf)
