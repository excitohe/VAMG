# VAMG

**Work in progress.**

## Dataset

Please first download the Sub-URMP dataset and organize it as following structure:
```
data
│──Sub-URMP
│──chunk_train.txt
│──chunk_test.txt
│──image_train.txt
│──image_test.txt
```

## Train & Evaluate

Easy for training, reset GUP ID by modify '--gpus' in main.py

```
python main.py
```

No support test script, please implement by yourself.