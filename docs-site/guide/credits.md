# Credits & License

## Model Credits

Ultra Fast Image Gen leverages several state-of-the-art open-weight diffusion models and quantization techniques. Full credit goes to the original authors and organizations:

- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** by Black Forest Labs
- **[Z-Image](https://github.com/Tongyi-MAI/Z-Image)** by Alibaba (Tongyi MAI)
- **[Anima](https://huggingface.co/circlestone-labs/Anima)** by Circlestone Labs
- **[Bonsai Image 4B](https://github.com/PrismML-Eng/image-studio)** by PrismML (via `prism-image-studio` + `mflux-prism`, built on [mflux](https://github.com/filipstrand/mflux) / [mlx](https://github.com/ml-explore/mlx))
- **[SDNQ Quantization](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic)** by Disty0
- **[Int8 Quantization](https://huggingface.co/aydin99/FLUX.2-klein-4B-int8)** using `optimum-quanto`
- **[Uncensored Text Encoder](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder)** by ponpoke

## License

See the original model licenses for specific usage terms and restrictions. 

- **FLUX.2:** Subject to Black Forest Labs' non-commercial/research license or commercial agreements.
- **Z-Image:** Subject to Alibaba's specific model license.
- **Anima / Bonsai:** Subject to their respective Hugging Face repository licenses.

This repository provides the optimization, quantization, and serving infrastructure. Users are responsible for complying with the underlying model licenses.
