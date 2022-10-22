Recently, diffusion models have been increasingly used in audio processing tasks including speech super-resolution.
Despite its strong performance in practice, prior works employ a rather simplistic sampling process throughout the generation.
In this paper, we introduce a novel sampling algorithm that leverages downsampled audio to condition the sampling of the diffusion.
We show that the proposed method acts as a drop-in replacement for the vanilla sampling process, and improves performance of the existing works.
By combining the proposed method with an unconditional diffusion model, we attain state-of-the-art results for speech super-resolusion on the VCTK Multi-Speaker benchmark.

## Animation of the Bandwidth Extension Process (200 steps, 12k to 48k)

![](ani/generation.gif)

## Samples: 24k to 48k, 50 steps

|            | p360_001                                                                           | p361_002                                                                           |
|------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
|            | ![](samples/x2/p360_001_mic1.png)                                                  | ![](samples/x2/p361_002_mic1.png)                                                  |
| Input      | <audio src="samples/x2/p360_001_mic1.wav" controls="" preload=""></audio>          | <audio src="samples/x2/p361_002_mic1.wav" controls="" preload=""></audio>          |
|            | ![](samples/origin/p360_001_mic1.png)                                              | ![](samples/origin/p361_002_mic1.png)                                              |
| Target     | <audio src="samples/origin/p360_001_mic1.wav" controls="" preload=""></audio>      | <audio src="samples/origin/p361_002_mic1.wav" controls="" preload=""></audio>      |
|            | ![](samples/x2-nuwave/p360_001_mic1.png)                                           | ![](samples/x2-nuwave/p361_002_mic1.png)                                           |
| NU-Wave    | <audio src="samples/x2-nuwave/p360_001_mic1.wav" controls="" preload=""></audio>   | <audio src="samples/x2-nuwave/p361_002_mic1.wav" controls="" preload=""></audio>   |
|            | ![](samples/x2-nuwave+/p360_001_mic1.png)                                          | ![](samples/x2-nuwave+/p361_002_mic1.png)                                          |
| NU-Wave+   | <audio src="samples/x2-nuwave+/p360_001_mic1.wav" controls="" preload=""></audio>  | <audio src="samples/x2-nuwave+/p361_002_mic1.wav" controls="" preload=""></audio>  |
|            | ![](samples/x2-nuwave2/p360_001_mic1.png)                                          | ![](samples/x2-nuwave2/p361_002_mic1.png)                                          |
| NU-Wave 2  | <audio src="samples/x2-nuwave2/p360_001_mic1.wav" controls="" preload=""></audio>  | <audio src="samples/x2-nuwave2/p361_002_mic1.wav" controls="" preload=""></audio>  |
|            | ![](samples/x2-nuwave2+/p360_001_mic1.png)                                         | ![](samples/x2-nuwave2+/p361_002_mic1.png)                                         |
| NU-Wave 2+ | <audio src="samples/x2-nuwave2+/p360_001_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x2-nuwave2+/p361_002_mic1.wav" controls="" preload=""></audio> |
|            | ![](samples/x2-wsrglow/p360_001_mic1.png)                                          | ![](samples/x2-wsrglow/p361_002_mic1.png)                                          |
| WSRGlow    | <audio src="samples/x2-wsrglow/p360_001_mic1.wav" controls="" preload=""></audio>  | <audio src="samples/x2-wsrglow/p361_002_mic1.wav" controls="" preload=""></audio>  |
|            | ![](samples/x2-mcg/p360_001_mic1.png)                                              | ![](samples/x2-mcg/p361_002_mic1.png)                                              |
| UDM+       | <audio src="samples/x2-mcg/p360_001_mic1.wav" controls="" preload=""></audio>      | <audio src="samples/x2-mcg/p361_002_mic1.wav" controls="" preload=""></audio>      |



## Samples: 16k to 48k, 50 steps

|            | p363_004                                                                           | p364_005                                                                           |
|------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
|            | ![](samples/x3/p363_004_mic1.png)                                                  | ![](samples/x3/p364_005_mic1.png)                                                  |
| Input      | <audio src="samples/x3/p363_004_mic1.wav" controls="" preload=""></audio>          | <audio src="samples/x3/p364_005_mic1.wav" controls="" preload=""></audio>          |
|            | ![](samples/origin/p363_004_mic1.png)                                              | ![](samples/origin/p364_005_mic1.png)                                              |
| Target     | <audio src="samples/origin/p363_004_mic1.wav" controls="" preload=""></audio>      | <audio src="samples/origin/p364_005_mic1.wav" controls="" preload=""></audio>      |
|            | ![](samples/x3-nuwave/p363_004_mic1.png)                                           | ![](samples/x3-nuwave/p364_005_mic1.png)                                           |
| NU-Wave    | <audio src="samples/x3-nuwave/p363_004_mic1.wav" controls="" preload=""></audio>   | <audio src="samples/x3-nuwave/p364_005_mic1.wav" controls="" preload=""></audio>   |
|            | ![](samples/x3-nuwave+/p363_004_mic1.png)                                          | ![](samples/x3-nuwave+/p364_005_mic1.png)                                          |
| NU-Wave+   | <audio src="samples/x3-nuwave+/p363_004_mic1.wav" controls="" preload=""></audio>  | <audio src="samples/x3-nuwave+/p364_005_mic1.wav" controls="" preload=""></audio>  |
|            | ![](samples/x3-nuwave2/p363_004_mic1.png)                                          | ![](samples/x3-nuwave2/p364_005_mic1.png)                                          |
| NU-Wave 2  | <audio src="samples/x3-nuwave2/p363_004_mic1.wav" controls="" preload=""></audio>  | <audio src="samples/x3-nuwave2/p364_005_mic1.wav" controls="" preload=""></audio>  |
|            | ![](samples/x3-nuwave2+/p363_004_mic1.png)                                         | ![](samples/x3-nuwave2+/p364_005_mic1.png)                                         |
| NU-Wave 2+ | <audio src="samples/x3-nuwave2+/p363_004_mic1.wav" controls="" preload=""></audio> | <audio src="samples/x3-nuwave2+/p364_005_mic1.wav" controls="" preload=""></audio> |
|            | ![](samples/x3-wsrglow/p363_004_mic1.png)                                          | ![](samples/x3-wsrglow/p364_005_mic1.png)                                          |
| WSRGlow    | <audio src="samples/x3-wsrglow/p363_004_mic1.wav" controls="" preload=""></audio>  | <audio src="samples/x3-wsrglow/p364_005_mic1.wav" controls="" preload=""></audio>  |
|            | ![](samples/x3-mcg/p363_004_mic1.png)                                              | ![](samples/x3-mcg/p364_005_mic1.png)                                              |
| UDM+       | <audio src="samples/x3-mcg/p363_004_mic1.wav" controls="" preload=""></audio>      | <audio src="samples/x3-mcg/p364_005_mic1.wav" controls="" preload=""></audio>      |


## Samples: 8k to 16k, 50 steps

|                    | p374_012                                                                                      | p376_233                                                                                      |
|--------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
|                    | ![](samples/16k/x2/p374_012_mic1.png)                                                         | ![](samples/16k/x2/p376_233_mic1.png)                                                         |
| Input              | <audio src="samples/16k/x2/p374_012_mic1.wav" controls="" preload=""></audio>                 | <audio src="samples/16k/x2/p376_233_mic1.wav" controls="" preload=""></audio>                 |
|                    | ![](samples/16k/origin/p374_012_mic1.png)                                                     | ![](samples/16k/origin/p376_233_mic1.png)                                                     |
| Target             | <audio src="samples/16k/origin/p374_012_mic1.wav" controls="" preload=""></audio>             | <audio src="samples/16k/origin/p376_233_mic1.wav" controls="" preload=""></audio>             |
|                    | ![](samples/16k/x2-nuwave2/p374_012_mic1.png)                                                 | ![](samples/16k/x2-nuwave2/p376_233_mic1.png)                                                 |
| NU-Wave 2          | <audio src="samples/16k/x2-nuwave2/p374_012_mic1.wav" controls="" preload=""></audio>         | <audio src="samples/16k/x2-nuwave2/p376_233_mic1.wav" controls="" preload=""></audio>         |
|                    | ![](samples/16k/x2-nuwave2+/p374_012_mic1.png)                                                | ![](samples/16k/x2-nuwave2+/p376_233_mic1.png)                                                |
| NU-Wave 2+         | <audio src="samples/16k/x2-nuwave2+/p374_012_mic1.wav" controls="" preload=""></audio>        | <audio src="samples/16k/x2-nuwave2+/p376_233_mic1.wav" controls="" preload=""></audio>        |
|                    | ![](samples/16k/x2-nuwave2-inpaint/p374_012_mic1.png)                                         | ![](samples/16k/x2-nuwave2-inpaint/p376_233_mic1.png)                                         |
| NU-Wave 2+ w/o MCG | <audio src="samples/16k/x2-nuwave2-inpaint/p374_012_mic1.wav" controls="" preload=""></audio> | <audio src="samples/16k/x2-nuwave2-inpaint/p376_233_mic1.wav" controls="" preload=""></audio> |
|                    | ![](samples/16k/x2-nvsr/p374_012_mic1.png)                                                    | ![](samples/16k/x2-nvsr/p376_233_mic1.png)                                                    |
| NVSR               | <audio src="samples/16k/x2-nvsr/p374_012_mic1.wav" controls="" preload=""></audio>            | <audio src="samples/16k/x2-nvsr/p376_233_mic1.wav" controls="" preload=""></audio>            |
|                    | ![](samples/16k/x2-mcg/p374_012_mic1.png)                                                     | ![](samples/16k/x2-mcg/p376_233_mic1.png)                                                     |
| UDM+               | <audio src="samples/16k/x2-mcg/p374_012_mic1.wav" controls="" preload=""></audio>             | <audio src="samples/16k/x2-mcg/p376_233_mic1.wav" controls="" preload=""></audio>             |
|                    | ![](samples/16k/x2-inpaint/p374_012_mic1.png)                                                 | ![](samples/16k/x2-inpaint/p376_233_mic1.png)                                                 |
| UDM+ w/o MCG       | <audio src="samples/16k/x2-inpaint/p374_012_mic1.wav" controls="" preload=""></audio>         | <audio src="samples/16k/x2-inpaint/p376_233_mic1.wav" controls="" preload=""></audio>         |
