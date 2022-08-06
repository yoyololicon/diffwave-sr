Current successful audio suepr-resolution models are based on supervised training, where a paired of input and output is given as guidance. 
Despite its strong performance in practice, these methods cannot generalize to data generated outside their training settings, such as a fixed upscaling rate or a range of input sampling rates. 
In this work, we leverage the recent success of diffusion models on solving inverse problems and introduce a new inference algorithm for diffusion models to do audio super-resolution. 
Coupling with a single unconditional audio generation model, our method can generate high quality 48 kHz audio from various input sampling rates. 
Evaluation on VCTK multi-speaker benchmark shows state-of-the-art results.

## Animation of the Bandwidth Extension Process (200 steps, 12k to 48k)

![](ani/generation.gif)

## Samples: 24k to 48k, 50 steps

| Input | Target | NU-Wave | NU-Wave+ | WSRGlow | Ours |
| ----- | ------ | ------- | -------- | ------- | ---- |
| <audio src="samples/x2/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave+/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-wsrglow/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-mcg/p360_001_mic1.wav" controls="" preload=""></audio> |
| <audio src="samples/x2/p361_002_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p361_002_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave/p361_002_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave+/p361_002_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-wsrglow/p361_002_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-mcg/p361_002_mic1.wav" controls="" preload=""></audio> |
| <audio src="samples/x2/p362_003_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p362_003_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave/p362_003_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave+/p362_003_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-wsrglow/p362_003_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-mcg/p362_003_mic1.wav" controls="" preload=""></audio> |

## Samples: 16k to 48k, 50 steps

| Input | Target | NU-Wave | NU-Wave+ | WSRGlow | Ours |
| ----- | ------ | ------- | -------- | ------- | ---- |
| <audio src="samples/x3/p363_004_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p363_004_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-nuwave/p363_004_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-nuwave+/p363_004_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-wsrglow/p363_004_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-mcg/p363_004_mic1.wav" controls="" preload=""></audio> |
| <audio src="samples/x3/p364_005_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p364_005_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-nuwave/p364_005_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-nuwave+/p364_005_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-wsrglow/p364_005_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-mcg/p364_005_mic1.wav" controls="" preload=""></audio> |
| <audio src="samples/x3/p374_006_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p374_006_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-nuwave/p374_006_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-nuwave+/p374_006_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-wsrglow/p374_006_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-mcg/p374_006_mic1.wav" controls="" preload=""></audio> |


## Samples: 12k to 48k, 200 steps

| Input | Target | WSRGlow | Ours |
| ----- | ------ | ------- | ---- |
| <audio src="samples/x4/p376_007_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/p376_007_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x4-wsrglow/p376_007_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x4-mcg-T200/p376_007_mic1.wav" controls="" preload=""></audio> |
| <audio src="samples/x4/s5_008_mic1.wav" controls="" preload=""></audio> | <audio src="samples/origin/s5_008_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x4-wsrglow/s5_008_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x4-mcg-T200/s5_008_mic1.wav" controls="" preload=""></audio> |