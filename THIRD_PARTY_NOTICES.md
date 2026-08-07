# Third-Party Notices

Unless otherwise noted, original TBSM code is licensed under the repository's [Apache License 2.0](./LICENSE). The adapted components below retain their upstream licenses and have been modified for TBSM.

| Component | Local scope | License |
|---|---|---|
| [PixelDiT](https://github.com/NVlabs/PixelDiT) | `imgnet/networks/backbones/pixeldit/{layers.py,model.py,wrapper.py}` | [NVIDIA NSCLv1](./LICENSES/PixelDiT-NSCLv1.txt) |
| [FD-Loss](https://github.com/Jiawei-Yang/FD-Loss) | adapted portions of `imgnet/metrics/{eval_all_fds.py,frechet_distance/**,utils/**}` | [MIT](./LICENSES/MIT-NOTICES.txt) |
| [JiT](https://github.com/LTH14/JiT) | `imgnet/networks/backbones/jit/{adapter.py,denoiser.py,layers.py,model.py}` | [MIT](./LICENSES/MIT-NOTICES.txt) |
| [EVA-02](https://github.com/baaivision/EVA) / [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) | `imgnet/networks/backbones/dit/pos_embed.py` | [MIT](./LICENSES/MIT-NOTICES.txt) |
| [UCGM](https://github.com/LINs-lab/UCGM) | `imgnet/networks/backbones/dit/{adapter.py,model.py}`, adapted from its TiT implementation | [Apache-2.0](./LICENSE) |
| [Meta DINOv2](https://github.com/facebookresearch/dinov2) | `imgnet/networks/backbones/dit/swiglu_ffn.py` | [Apache-2.0](./LICENSE) |
| [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | `center_crop_arr` in `imgnet/networks/representors/utils.py`; reference implementation for `compute_fid` in `imgnet/metrics/frechet_distance/metrics.py` | [MIT](./LICENSES/MIT-NOTICES.txt) |

The PixelDiT files above and their modifications are governed by NSCLv1, not Apache-2.0. NSCLv1 permits their use only for non-commercial research or evaluation. Copyright and permission notices for MIT-licensed components are consolidated in [MIT-NOTICES.txt](./LICENSES/MIT-NOTICES.txt).

External model weights, datasets, and separately installed dependencies remain subject to their respective upstream terms.
