# Very Deep VAEs (VD-VAE) [Work in Progress]

> Very much a work in progress and mostly untested. Patience people!

PyTorch implementation of Very Deep VAE (VD-VAE) from the paper "Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images"

Based off OpenAI implementation found [here](https://github.com/openai/vdvae) but removing clutter (paper visualizations, parallel compute, large datasets) to focus purely on the underlying architecture.

Original paper can be found [here](https://arxiv.org/abs/2011.10650).

VD-VAEs are quite memory hungry and currently this program does not support multiple GPUs. Tweak the `hps.py` file until it runs on your GPU.

## Modifications
- Replacing default residual connections with ReZero connections (see citations). Might enable faster convergence at larger depths.
- Replacing Mixture of Logistics output and loss with simple MSE reconstruction and KL loss, more weighted towards reconstruction loss.
    - May implement MoL in the future, when experimenting on more complex distributions

## Task List
- [x] Basic architecture
- [X] Training script
- [ ] **Sampling functions**
- [X] Weight / bias initialisation (as in paper)
- [ ] Gradient skipping (as in paper)
- [ ] Explore further modifications

### Citations
```
@misc{child2020deep,
      title={Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images}, 
      author={Rewon Child},
      year={2020},
      eprint={2011.10650},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{bachlechner2020rezero,
      title={ReZero is All You Need: Fast Convergence at Large Depth}, 
      author={Thomas Bachlechner and Bodhisattwa Prasad Majumder and Huanru Henry Mao and Garrison W. Cottrell and Julian McAuley},
      year={2020},
      eprint={2003.04887},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
