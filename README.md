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
Upcoming...

## Citation
If you find this repository useful, please cite our paper:
```
@article{zhou2022conner,
  title={ConNER: Consistency Training for Cross-lingual Named Entity Recognition},
  author={Zhou, Ran and Li, Xin and Bing, Lidong and Cambria, Erik and Si, Luo and Miao, Chunyan},
  journal={arXiv preprint arXiv:2211.09394},
  year={2022}
}
```

## Acknowledgements
The training code in this repo has been built upon [pytorch-neural-crf](https://github.com/allanj/pytorch_neural_crf)
