# Latent Diffusion

Welcome to this repository explaining Latent Diffusion Models! In the immediate directory, you'll find [Schedulers_ML.md](https://github.com/ejohansson13/Latent-Diffusion/blob/main/Schedulers_ML.md) and [StableDiffusion_ML.md](https://github.com/ejohansson13/Latent-Diffusion/blob/main/StableDiffusion_ML.md). Both assume a baseline of prior machine learning experience and aim to explain concepts directly related to Latent Diffusion Models. Schedulers_ML.md focuses on the important algorithmic advancements to scheduling in diffusion through the lens of four key research papers. It touches on some of the core equations within those algorithms, but favors conceptual clarity over mathematical rigor. StableDiffusion_ML.md examines the development of Stable Diffusion, one of the most popular open-source machine learning models globally, from its original research paper. It is intended to explain, with a high degree of detail, the design and functionality of a text-to-image synthesis model. [Examples.md](https://github.com/ejohansson13/Latent-Diffusion/blob/main/Examples.md) examines the subsequent results of building a Latent Diffusion Model from scratch and the diversity of images that are able to be successfully generated. All code employed in the Examples.md folder can be found in the [code](https://github.com/ejohansson13/Latent-Diffusion/tree/main/Code) folder, while pertinent model weights can be found below. Thanks!

### Configure model workspace

1. Create environment with the requirements.txt file by running:
```
pip install -r requirements.txt
```

2. Download ```vocab.json``` and ```merges.txt``` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer and save them in the data folder

3. Download ```v1-5-pruned-emaonly.ckpt``` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in the data folder
