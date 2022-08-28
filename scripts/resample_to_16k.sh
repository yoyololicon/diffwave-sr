#!/bin/bash

src="$1"
target="$2"

for f in "$src"/*/*mic1.wav
do
    echo "$f"
    python -c "import soundfile as sf; import samplerate; import os; x, sr = sf.read('$f'); x_hat = samplerate.resample(x, 16000 / sr, 'sinc_best'); output_name = '$target/$(basename "$(dirname $f)")/$(basename $f)'; os.makedirs(os.path.dirname(output_name), exist_ok=True); sf.write(output_name, x_hat, 16000)"
done