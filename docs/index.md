## Inpainting in the Frequency - Universal Speech Super-Resolution with Diffusion Models

Chin-Yun Yu, Sung-Lin Yeh

### Abstract

Current successful audio suepr-resolution models are based
on supervised training, where a paired of input and output is
given as guidance. Despite its strong performance in practice, these methods cannot generalize to data generated outside their training settings, such as a fixed upscaling rate or a
range of input sampling rates. In this work, we leverage the
recent success of diffusion models on solving inverse problems and introduce a new inference algorithm for diffusion
models to do audio super-resolution. Coupling with a single
unconditional audio generation model, our method can generate high quality 48 kHz audio from various input sampling
rates. Evaluation on VCTK multi-speaker benchmark shows
state-of-the-art results.

### Animation of the Inpainting Process (200 steps, 12k to 48k)

![](ani/generation.gif)

You can use the [editor on GitHub](https://github.com/yoyololicon/diffwave-sr/edit/main/docs/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/yoyololicon/diffwave-sr/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
