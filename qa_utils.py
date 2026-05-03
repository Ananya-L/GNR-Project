import os
import re
from pathlib import Path
import torch
import numpy as np
from PIL import Image


_ANSWERER = None
torch.backends.cuda.matmul.allow_tf32 = True

def _parse_option_number(text):
    """Extract one valid MCQ option from a model response."""
    if text is None:
        return 5

    text = str(text).strip()
    exact = re.fullmatch(r"[1-5]", text)
    if exact:
        return int(text)

    patterns = [
        r"\boption\s*([1-5])\b",
        r"\banswer\s*(?:is|:)?\s*([1-5])\b",
        r"\b([1-5])\b",
    ]
    found = []
    for pattern in patterns:
        found.extend(int(match) for match in re.findall(pattern, text, flags=re.I))

    found = [value for value in found if 1 <= value <= 5]
    return found[0] if found else 5

class MapQuestionAnswerer:
    def __init__(self, model_dir=None):
        self.project_dir = Path.cwd()
        env_model_dir = os.environ.get("GNR_QA_MODEL_DIR")
        self.model_dir = Path(model_dir or env_model_dir) if (model_dir or env_model_dir) else self._find_model_dir()
        self.model = None
        self.processor = None
        self.process_vision_info = None

    def _find_model_dir(self):
        candidates = [
            self.project_dir / "models" / "Qwen2-VL-2B-Instruct",
            self.project_dir.parent / "models" / "Qwen2-VL-2B-Instruct",
            self.project_dir / "GNR_Project" / "models" / "Qwen2-VL-2B-Instruct",
        ]
        for candidate in candidates:
            if (candidate / "config.json").exists():
                return candidate
        return candidates[0]

    def _load(self):
        if self.model is not None:
            return

        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Qwen2-VL dependencies are not available. Install requirements.txt "
                "or leave uncertain answers as 5."
            ) from exc

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Local model folder not found: {self.model_dir}")

        if torch.cuda.is_available():
            device_map = "auto"
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1536 * 1536,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.process_vision_info = process_vision_info

    def answer(self, question, options, stitched_image):
        if os.environ.get("GNR_USE_VLM", "1") == "0":
            return 5

        self._load()
        image = self._to_pil(stitched_image)


        prompt = self._build_prompt(question, options)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = self.process_vision_info(messages)
        target_device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(target_device)

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False
                )
        except RuntimeError:
            # 🔥 fallback if memory issue
            image.thumbnail((1344, 1344))

            image_inputs, video_inputs = self.process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(target_device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False
                )

        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        ans = _parse_option_number(response)

        # basic safety
        if ans not in [1, 2, 3, 4]:
            return 5

        return ans

    @staticmethod
    def _to_pil(stitched_image):
        if isinstance(stitched_image, Image.Image):
            return stitched_image.convert("RGB")
        if isinstance(stitched_image, (str, Path)):
            return Image.open(stitched_image).convert("RGB")
        if isinstance(stitched_image, np.ndarray):
            return Image.fromarray(stitched_image.astype(np.uint8)).convert("RGB")
        raise TypeError(f"Unsupported stitched_image type: {type(stitched_image)!r}")

    @staticmethod
    def _build_prompt(question, options):
        options_text = "\n".join(f"{idx}. {option}" for idx, option in enumerate(options, 1))
        return (
    "You are answering a multiple-choice question using a map image.\n"
    "Use visible labels and approximate spatial reasoning.\n"
    "For directions (north, south, etc.), use relative positions on the image.\n"
    "For proximity (near, closest), choose the visually nearest labeled location.\n"
    "Do not invent names not present in the options.\n"
    "If a clear answer is visible, choose it confidently.\n"
    "If none of the options match the image, return 5.\n\n"
    f"Question: {question}\n"
    f"Options:\n{options_text}\n"
    "Return exactly one number: 1, 2, 3, 4, or 5."
)


def get_answerer():
    global _ANSWERER
    if _ANSWERER is None:
        _ANSWERER = MapQuestionAnswerer()
    return _ANSWERER


def clear_qa_cache():
    global _ANSWERER
    _ANSWERER = None


def answer_question(question, options, stitched_image):
    return get_answerer().answer(question, options, stitched_image)
