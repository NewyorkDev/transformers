# Learning Folder for Mistral Fine-tuning

This folder is designed for collecting human-written text samples that can be used to fine-tune the Mistral language model.

## How to Use

1. Drop your text files (.txt) in this folder
2. Each file should contain high-quality human-written text
3. Run the `mistral_finetune.py` script and point it to this folder when prompted

## How Much Text is Needed?

For effective fine-tuning:

- **Minimum recommendation**: 10-20 text samples
- **Ideal file size**: 1000-2000 words per file (roughly 5-10KB each)
- **Content quality**: High-quality, well-written text that represents the style you want the model to learn
- **Variety**: Include different examples of the writing style you want to emulate

The more high-quality examples you provide, the better the model will learn your desired writing style. However, even a few well-written samples can make a noticeable difference in the model's output.

## Tips for Best Results

- Use consistent formatting across your text files
- Remove any headers, footers, or metadata that you don't want the model to learn
- Consider organizing text by topics if you want the model to learn specific subject matter
- Make sure the text is representative of the writing style you want to generate

The `mistral_finetune.py` script will automatically use these files for training when you select this folder as your dataset source.