# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input
import os
import time
import torch
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "xtuner/llava-phi-3-mini-hf"
MODEL_CACHE = "checkpoints"
WEIGHTS_URL = "https://weights.replicate.delivery/default/xtuner/llava-phi-3-mini-hf/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(WEIGHTS_URL, MODEL_CACHE)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="What are these?"),
        max_new_tokens: int = Input(description="Max new tokens", default=200, ge=8, le=4096)
    ) -> str:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")
        prompt_format = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
        inputs = self.processor(prompt_format, img, return_tensors='pt').to('cuda')
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        result = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return result