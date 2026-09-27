# Attribution for the audio in this repository

The speech clips in `examples/` and `tests/fixtures/` come from two public corpora, both
licensed CC BY 4.0, which requires naming the source and stating that the files were modified.
This page is that notice. Every clip was converted to 16 kHz mono 16-bit wav; `tokens.json` in
each example folder names the source clip of every file.

| Files | Source | Licence | Changes |
|---|---|---|---|
| `examples/01_parallel_speakers/p230.wav`, `p244.wav`, `p245.wav`, `p259.wav`; `tests/fixtures/speech_vctk_p230_001.wav` | CSTR VCTK Corpus v0.92, clips `p230_001_mic1`, `p244_001_mic1`, `p245_001_mic1`, `p259_001_mic1`; Yamagishi, Veaux & MacDonald, University of Edinburgh, 2019 — [datashare.ed.ac.uk/handle/10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | converted to 16 kHz mono wav |
| `examples/02_augmentations/*.wav` | CSTR VCTK Corpus v0.92, clip `p360_418_mic1`, as above | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | converted to 16 kHz mono wav; four versions add numerically generated distortions: band-passed Gaussian noise at 5 dB SNR, brown noise at 2 dB SNR, a broadcast EQ curve, a +8 semitone pitch shift |
| `tests/fixtures/speech_librispeech_3081_166546_0002.wav` | LibriSpeech ASR corpus, dev-clean, clip `3081-166546-0002`, © 2014 Vassil Panayotov — [openslr.org/12](https://www.openslr.org/12/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | decoded from flac to 16 kHz mono wav |
| `tests/fixtures/synth_*.wav` | generated numerically (amplitude-modulated tone, click train, low-pass noise, sine sweep); no recording involved | MIT, with this repository | none |

The files derived from these clips — `tokens.json` and the `.svg` renderings — are released
under this repository's MIT licence; for files derived from a CC BY clip, the attribution above
carries over.
