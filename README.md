### Title: Image Generation Using FLUX.1-dev and FLUX.1-Turbo-Alpha

### Description:
This repository includes scripts for generating high-quality images on CPU or GPU using **FLUX.1-dev** and **FLUX.1-Turbo-Alpha** models by **AlimamaCreative Team**. Both models excel at text-to-image tasks, with **Turbo-Alpha** optimized for faster generation while maintaining quality.

### Key Features:
- **FLUX.1-dev**: Ideal for detailed text-to-image generation, offering high fidelity with multi-head discriminators.
- **FLUX.1-Turbo-Alpha**: Focuses on speed, generating high-resolution images quickly with minimal quality trade-offs.
  
Both models generate 1024x1024 images, enabling efficient creative workflows such as **advertising** and **content creation**.

### Usage:
- **Text-to-Image Generation**: Converts prompts into high-quality images within seconds.

<div style="background-color: white; border-radius: 10px; padding: 10px; display: inline-block;">
  <img src="./images/T2I.png" alt="T2I Example" style="border-radius: 10px;">
</div>

### Technical Details:
- **Training**: 1M high-quality images, BF16 precision, 1024x1024 resolution.
- **Guidance Scale**: Optimal results at `guidance_scale=3.5`, ensuring prompt adherence.
