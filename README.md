# Continuous-Time Linear Positional Embedding for Irregular Time Series Forecasting


This is the Pytorch implementation of CTLPE in the following paper: 
[Continuous-Time Linear Positional Embedding for Irregular Time Series Forecasting](https://arxiv.org/pdf/2409.20092), on the model [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/pdf/2012.07436).



<p align="center">
<img src=".\img\CTLPE.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> The overview of CTLPE.
</p>

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Usage

Commands for training and testing the model on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

# ETTh2
python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

# ETTm1
python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t
```

More parameter information please refer to `main_informer.py`.

