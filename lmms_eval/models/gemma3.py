import os
import warnings
import numpy as np
from typing import Tuple, List, Optional, Union
import PIL
import torch

from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

@register_model("gemma3")
class Gemma3(lmms):
    """
    Gemma3 model from https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d
    """
    def __init__(
        self,
        pretrained: str = "google/gemma-3-4b-it",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        device_map: str = "auto",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_length: Optional[int] = 8192,
        max_image_size: Optional[int] = None,
        use_flash_attention_2: Optional[bool] = False,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs,
        ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        attn_implementation = "eager"
        if use_flash_attention_2:
            attn_implementation = "flash_attention_2"
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained,
            device_map=self.device_map,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            use_fast=True,
            padding_side="left",
            trust_remote_code=trust_remote_code
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._tokenizer.padding_side = "left"
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt

        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], "Unsupported distributed type provided. Only DDP and FSDP are supported."

            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator
    @property
    def config(self):
        return self._config
    
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Gemma3")

    def generate_until_multi_round(self, requests: List[Instance]) ->List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for Gemma3")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_cords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_cords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                
            # messages = [{"role":"system", "content": self.system_prompt}, {"role": "user", "content": []}]
            messages = [{"role": "user", "content": []}]
            images = []
            for visual in visual_list[i]:
                if isinstance(visual, Image.Image):
                    images.append(visual)
                else:
                    raise ValueError("We only support PIL.Image for now.")
            for _ in range(len(images)):
                messages[-1]["content"].append({"type": "image"})
            for cont in contexts:
                messages[-1]["content"].append({"type": "text", "text": cont})
            prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(images, prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            # gen_kwargs adjust
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    top_p=gen_kwargs["top_p"],
                    use_cache=self.use_cache,
                    num_beams=gen_kwargs["num_beams"],
                )
                output = output[:, inputs["input_ids"].shape[-1]:]
                output_text = self._processor.decode(output[0], skip_special_tokens=True)
                res.append(output_text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), output_text)
            pbar.update(1)
        pbar.close()
        return res