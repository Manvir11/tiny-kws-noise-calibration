# Third-party sources

| Component | Use in this study | Source and license |
| --- | --- | --- |
| MLPerf Tiny | Frozen DS-CNN checkpoint and reference front-end and conversion workflow. The study adapts the front end for batched features and noise mixtures. | [Keyword spotting implementation at the pinned commit](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting). [Apache License 2.0](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/LICENSE.md). |
| Speech Commands v0.02 | Speech utterances and the official speaker split manifests. | Pete Warden, [*Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*](https://arxiv.org/abs/1804.03209). [Public archive](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). |
| DEMAND | Channel 1 of designated environmental recordings at 16 kHz. | Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent, [DEMAND dataset](https://zenodo.org/records/1227121). [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). |

The study scripts verify and extract audio locally. They combine selected
speech and noise in memory to derive features. The repository contains
the study code and selected analysis outputs. Obtain third-party audio and
the checkpoint directly from their sources and follow the corresponding terms.
