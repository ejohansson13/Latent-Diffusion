# Latent Diffusion

All of the architecture in the attached [code](https://github.com/ejohansson13/Latent-Diffusion/tree/main/Code) folder is the same as the architecture described in the [Stable Diffusion](https://github.com/ejohansson13/Latent-Diffusion/blob/main/StableDiffusion_ML.md) page. The schedulers designed for compatibility with this model are also the schedulers described in the [schedulers](https://github.com/ejohansson13/Latent-Diffusion/blob/main/Schedulers_ML.md) page.

The four schedulers provided for this PyTorch-based LDM are: DDPM, DDIM, Euler, and Euler Ancestral. All four are capable of generating images following the user-provided prompt, although the performance of the deterministic Euler algorithm falls behind the other three. The model weights employed for these architectures are v1.5 model weights and the [fidelity of text-generated images has significantly improved since the first generation](https://arxiv.org/pdf/2403.03206#page=11). Schedulers' performance varies dependent on the provided prompt and number of inference steps. Let's take a look at some examples.

### "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."

| DDPM          | DDIM          |  Euler        |Euler Ancestral|
| :-----------: | :-----------: | :-----------: | :-----------: |
| ![](Images/examples/ddpm_cat.png) | ![](Images/examples/ddim_cat.png) | ![](Images/examples/euler_cat.png) | ![](Images/examples/euler_a_cat.png) |

Some thoughts.

### "An astronaut floating in space, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."

| DDPM          | DDIM          |  Euler        |Euler Ancestral|
| :-----------: | :-----------: | :-----------: | :-----------: |
| ![](Images/examples/ddpm_astronaut_50.png) | ![](Images/examples/ddim_astronaut_50.png) | ![](image here) | ![](Images/examples/euler_a_astronaut_50.png) |




| DDPM          | DDIM          |  Euler        |Euler Ancestral|
| :-----------: | :-----------: | :-----------: | :-----------: |
| ![](Images/examples/ddpm_astronaut_75.png) | ![](Images/examples/ddim_astronaut_75.png) | ![](image here) | ![](Images/examples/euler_a_astronaut_75.png) |

Some thoughts.

### "A zombie in the style of Picasso."

| DDPM          | DDIM          |  Euler        |Euler Ancestral|
| :-----------: | :-----------: | :-----------: | :-----------: |
| ![](Images/examples/ddpm_zombie_25.png) | ![](Images/examples/ddim_zombie_25.png) | ![](image here) | ![](Images/examples/euler_a_zombie_25.png) |



| DDPM          | DDIM          |  Euler        |Euler Ancestral|
| :-----------: | :-----------: | :-----------: | :-----------: |
| ![](Images/examples/ddpm_zombie_50.png) | ![](Images/examples/ddim_zombie_50.png) | ![](image here) | ![](Images/examples/euler_a_zombie_50.png) |




| DDPM          | DDIM          |  Euler        |Euler Ancestral|
| :-----------: | :-----------: | :-----------: | :-----------: |
| ![](Images/examples/ddpm_zombie_75.png) | ![](Images/examples/ddim_zombie_75.png) | ![](image here) | ![](Images/examples/euler_a_zombie_75.png) |


Some thoughts.

### "A Ferrari sports car, modern car, red color, highly detailed, ultra sharp, perfectly centered, cinematic, 100mm lens, 8k resolution."

### "A painting of a squirrel eating a burger."


Images output with the same prompt but different schedulers.

Disconnect with Euler A pictures -> not centered correctly. Argue that training data may have been cropped or off-center. Using model v1.5 weights. v2 weights likely resolve issue


Effect of increasing number of inference steps on image quality. Same prompt within the same scheduler.
