[EfficientNetV2](https://arxiv.org/abs/2104.00298) implementation using PyTorch

### Steps

* configure `imagenet` path by changing `data_dir` in `main.py`
* `bash ./main.sh $ --train` for training model, `$` is number of GPUs
* see `EfficientNet` class in `nets/nn.py` for different versions

### Note

* the default training configuration is for `EfficientNetV2-S`

### Parameters and FLOPS

* `python main.py --benchmark`

```
Number of parameters: 21458488
Time per operator type:
        1504.95 ms.    80.5982%. Conv
        225.509 ms.    12.0772%. Sigmoid
        115.112 ms.     6.1649%. Mul
        12.7341 ms.   0.681982%. Add
        7.50523 ms.   0.401946%. AveragePool
        1.40185 ms.  0.0750768%. FC
      0.0112697 ms. 0.000603555%. Flatten
        1867.22 ms in Total
FLOP per operator type:
        16.7287 GFLOP.     99.708%. Conv
      0.0412707 GFLOP.   0.245986%. Mul
     0.00516096 GFLOP.  0.0307609%. Add
       0.002561 GFLOP.  0.0152643%. FC
        16.7777 GFLOP in Total
Feature Memory Read per operator type:
        291.409 MB.    51.8224%. Mul
        224.497 MB.    39.9231%. Conv
        41.2877 MB.    7.34234%. Add
        5.12912 MB.   0.912131%. FC
        562.323 MB in Total
Feature Memory Written per operator type:
        165.083 MB.    50.2087%. Mul
        143.062 MB.    43.5114%. Conv
        20.6438 MB.    6.27867%. Add
          0.004 MB. 0.00121657%. FC
        328.793 MB in Total
Parameter Memory per operator type:
        79.9537 MB.    93.9773%. Conv
          5.124 MB.    6.02273%. FC
              0 MB.          0%. Add
              0 MB.          0%. Mul
        85.0777 MB in Total
```

### Results

* `python main.py --test` for trained model testing

|       name       | resolution | acc@1 | acc@5 | #params |  FLOPS  | resample | training loss |
|:----------------:|:----------:|:-----:|:-----:|:-------:|:-------:|---------:|--------------:|
| EfficientNetV2-S |  384x384   | 83.9  | 96.7  |  21.46  | 16.7777 | BILINEAR |  CrossEntropy |
| EfficientNetV2-S |  384x384   |   -   |   -   |  21.46  | 16.7777 | BILINEAR |      PolyLoss |
| EfficientNetV2-M |     -      |   -   |   -   |    -    |    -    |        - |             - |
| EfficientNetV2-L |     -      |   -   |   -   |    -    |    -    |        - |             - |
