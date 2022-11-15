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


## Extending to non-zero phase response lowpass filters

When using IIR low pass filter to downsample audio, it introduces non-linear phase delays, thus breaking the assumption that the frequency mask is real value.
An easy solution to compensate for the delays is **applying the same filter again during upsampling but in a backward direction of time**.
We conducted the same 48 kHz experiment in the paper agian but with a 8th order Chebyshev Type I low pass filter.

|            | 2x | 3x |
|------------|----|----|
| NU-Wave    | 0.8667916475732103   | 0.9955991584912919   |
| NU-Wave 2  |  |    |
| NU-Wave+   |  1.0255468693118877  | 1.3233148328982787   |
| NU-Wave 2+ |    |    |
| UDM+       | 0.6394651938466991   | 0.7940077889237551   |
