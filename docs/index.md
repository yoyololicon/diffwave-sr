## Inpainting in the Frequency - Universal Speech Super-Resolution with Diffusion Models

Chin-Yun Yu, Sung-Lin Yeh

### Abstract

Current successful audio suepr-resolution models are based
on supervised training, where a paired of input and output is
given as guidance. Despite its strong performance in practice, these methods cannot generalize to data generated outside their training settings, such as a fixed upscaling rate or a
range of input sampling rates. In this work, we leverage the
recent success of diffusion models on solving inverse problems and introduce a new inference algorithm for diffusion
models to do audio super-resolution. Coupling with a single
unconditional audio generation model, our method can generate high quality 48 kHz audio from various input sampling
rates. Evaluation on VCTK multi-speaker benchmark shows
state-of-the-art results.

### Animation of the Bandwidth Extension Process (200 steps, 12k to 48k)

![](ani/generation.gif)

### Samples: 24k to 48k, 50 steps

| Input | Target | NU-Wave | NU-Wave+ | WSRGlow | Ours |
| ----- | ------ | ------- | -------- | ------- | ---- |
| <audio src="samples/x2/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave+/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-wsrglow/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-mcg/p360_001_mic1.wav" controls="" preload=""></audio> |