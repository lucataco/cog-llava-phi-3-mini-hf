# xtuner/llava-phi-3-mini-hf Cog Model

This is an implementation of [xtuner/llava-phi-3-mini-hf](https://huggingface.co/xtuner/llava-phi-3-mini-hf) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

To run a prediction:

    cog predict -i image=@lava.jpg -i prompt="What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"

![lava](lava.jpg)

