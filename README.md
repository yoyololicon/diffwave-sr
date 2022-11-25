# Speech Super-resolution with Unconditional Diffwave
[![pages-build-deployment](https://github.com/yoyololicon/diffwave-sr/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/yoyololicon/diffwave-sr/actions/workflows/pages/pages-build-deployment)
[![arXiv](https://img.shields.io/badge/arXiv-2210.15793-00ff00.svg)](https://arxiv.org/abs/2210.15793)

Source code of the paper [**Conditioning and Sampling in Variational Diffusion Models for Speech Super-Resolution**](https://arxiv.org/abs/2210.15793).


## Training

1. Install python requirements.

```commandline
pip install requirements.txt
```

2. Please convert all the data files into `.wav` format and put them under the same directory. The following command will train a 48 kHz UDM.
```commandline
python train.py model.res_channels=64 epochs=50 sr=48000 train_T=0 dataset.size=120000 dataset.segment=32768 dataset.data_dir=/your/vctk/train/set/ loader.batch_size=12 scheduler.patience=1000000
```


## Evaluation

The numbers in the paper can be reproduced with following commands.

* `rate`: the upscaling ratio.
* `downsample-type`: the downsampling filter.
* `infer-type`: the upscaling method.
* `lr`: the $\eta$ value in the paper.

### Spline Interpolation

```commandline
python vctk_dsp_baseline.py /your/vctk/test/set/ --downsample-type sinc --infer-type spline --rate 2
```

### UDM+

```commandline
python -W ignore vctk_infer.py outputs/XXXX/saved/training_checkpoint_500000.pt outputs/XXXX/.hydra/config.yaml /your/vctk/test/set --rate 2 -T 50 --infer-type manifold --downsample-type stft --lr 0.67
```

### UDM+ without MCG

```commandline
python -W ignore vctk_infer.py outputs/XXXX/saved/training_checkpoint_500000.pt outputs/XXXX/.hydra/config.yaml /your/vctk/test/set --rate 3 -T 50 --infer-type inpainting --downsample-type sinc
```

### NU-Wave(+)

The checkpoint of UDM is used for noise scheduling.
For training NU-Wave, please refer to [here](https://github.com/mindslab-ai/nuwave). For evaluating NU-Wave+, change `infer-type` to `nuwave-manifold` and specify the value of `lr`.

```commandline
python -W ignore vctk_infer.py outputs/XXXX/saved/training_checkpoint_500000.pt outputs/XXXX/.hydra/config.yaml /your/vctk/test/set --nuwave-ckpt /XXXX/checkpoints_nuwave_x2/nuwave_x2_01_07_22_epoch\=645_EMA --rate 2 -T 50 --infer-type nuwave --downsample-type stft
```

### NU-Wave 2(+)

The checkpoint of UDM is used for noise scheduling.
For training NU-Wave 2, please refer to [here](https://github.com/mindslab-ai/nuwave2). For evaluating NU-Wave 2+, change `infer-type` to `nuwave2-manifold` and specify the value of `lr`.

```commandline
python -W ignore vctk_infer.py outputs/XXXX/saved/training_checkpoint_500000.pt outputs/XXXX/.hydra/config.yaml /your/vctk/test/set --nuwave-ckpt /XXXX/nuwave2_08_14_09_epoch\=72_EMA --rate 3 -T 50 --infer-type nuwave2 --downsample-type sinc
```

We'll release the script for evaluating WSRGlow and NVSR in the future.


## Pre-trained Checkpoints

* [48 kHz](ckpt/vctk_48k_udm/saved/training_checkpoint_500000.pt)
* [16 kHz](ckpt/vctk_16k_udm/saved/training_checkpoint_500000.pt)


## Errata

We found that we didn't apply the same upsampling method to the condition inputs of NU-Wave 2.
The original implementation used `scipy.signal.resample_poly`, but we used sinc interpolation instead.
This mismatch produces some artefacts around the Nyquist frequency in the outputs, which can be observed in the old demo page samples.
We report the correct experiment results with proper condition signal in the below tables.

### NU-Wave 2, 48 kHz

|            |    2x Sinc   | 2x STFT      | 3x Sinc      | 3x STFT      |
|------------|:------------:|:------------:|:------------:|:------------:|
| Nu-Wave 2  | 0.76 -> 0.75 | 0.75 -> 0.71 | 0.90 -> 0.89 | 0.89 -> 0.86 |
| Nu-Wave 2+ | 0.75 -> 0.74 | 0.74 -> 0.71 | 0.90 -> 0.89 | 0.90 -> 0.86 |


### NU-Wave 2, 8 kHz -> 16 kHz

|                    |    Sinc LSD  | STFT LSD     | Sinc PESQ    | STFT PESQ    |
|--------------------|:------------:|:------------:|:------------:|:------------:|
| Nu-Wave 2          | 1.13 -> 1.06 | 1.10 -> 0.94 | 3.30 -> 3.38 | 3.32 -> 3.33 |
| Nu-Wave 2+         | 1.06 -> 0.99 | 1.04 -> 0.90 | 3.25 -> 3.27 | 3.28 -> 3.24 |
| Nu-Wave 2+ w/o MCG | 1.12 -> 1.04 | 1.09 -> 0.94 | 3.44 -> 3.46 | 3.46 -> 3.39 |



## Extending to non-zero phase response lowpass filters

When using IIR lowpass filter to downsample audio, it introduces non-linear phase delays, thus breaking the assumption that the frequency mask is real value.
An easy solution to compensate for the delays is **applying the same filter again during upsampling but in a backward direction of time**.
We conducted the same 48 kHz experiment in the paper again but with a 8th order Chebyshev Type I lowpass filter.

|            | 2x   | 3x   |
|------------|:----:|:----:|
| NU-Wave    | 0.87 | 1.00 |
| NU-Wave 2  | 0.73 | 0.87 |
| NU-Wave+   | 1.03 | 1.32 |
| NU-Wave 2+ | 0.86 | 1.00 |
| UDM+       | 0.64 | 0.79 |
