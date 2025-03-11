import argparse

from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='auto')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # SFT merge
    # parser.add_argument("--model-base", type=str, default="checkpoints/Video-LLaVA-7B",)
    # parser.add_argument("--model-path", type=str, default="checkpoints/homiebot-7b-sft-lora")
    # parser.add_argument("--save-model-path", type=str, default="checkpoints/homiebot-7b-sft")

    # DPO merge
    parser.add_argument("--model-base", type=str, default="checkpoints/homiebot-7b-sft",)
    parser.add_argument("--model-path", type=str, default="checkpoints/homiebot-7b-dpo-lora")
    parser.add_argument("--save-model-path", type=str, default="checkpoints/homiebot-7b-dpo")
    
    
    args = parser.parse_args()

    merge_lora(args)
