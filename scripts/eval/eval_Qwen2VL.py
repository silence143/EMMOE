import re   
import os
import json
import socket
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import HomieBot


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "checkpoints/Qwen2-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("checkpoints/Qwen2-VL-7B-Instruct")

exp_dir = "exp"
exp_name = "Qwen2VL"
exp_path = os.path.join(exp_dir, exp_name)

homie = HomieBot()

split = "train"
split_path = f"EMMOE-100/data/{split}"

for task_idx in os.listdir(split_path):
    task_path = os.path.join(split_path, task_idx)
    info_path = os.path.join(task_path, "info.txt")
    save_dir = os.path.join(exp_path, f"{task_idx}")
        
    with open(info_path, 'r') as file:
        task = file.readline().split('Task:')[1].strip()

    for i in range(1, 4):
        save_path = os.path.join(save_dir, f"{i}")
        homie.conv.reset()
        homie.inventory = []
        feedback = ""
        historical_execution = ""

        while homie.conv.round < homie.conv.max_round:
            homie.conv.round += 1
            instruction = homie.generate_instruction(task, feedback, historical_execution)
            images = homie.comm.receive_env_images()
            print(images)
            
            match = None
            for i in range(5):
                messages = [
                    {"role": "system", "content": homie.conv.system,},
                    {"role": "user",
                     "content": "here is an example output, please strictly follow its format and system reminders in your output:\nAnalysis: According to my final task, I need to fetch apples first, but it's a better choice to go the fridge and open it first, which will avoid potential conflicts, so I should go to the fridge next\nSubtask: [Go to, fridge]\nModel: NoMaD\n",
                    },
                    {"role": "assistant",
                     "content": "I will surely follow the given format, now you can send prompt to me.",
                    },
                    {"role": "user",
                     "content": [
                            {"type": "image", "image": images[0]},
                            {"type": "image", "image": images[1]},
                            {"type": "image", "image": images[2]},
                            {"type": "image", "image": images[3]},
                            {"type": "text", "text": instruction},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                outputs = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                output = outputs[0]
                
                pattern = r'.*Analysis: *(.+?) *Subtask: *\[(.*?)\].*Model: *(.*?)$'
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    break
                pattern = r'.*Analysis: *(.+?) *Subtask: *(.*?) *Model: *(.*?)$'
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    break

                print(output)    
            if not match:
                homie.comm.send_subtask(f"End")
                homie.comm.receive_feedback()
                break

            homie.conv.history.append(f"USER:\n{instruction}ASSISTANT:\n{output}\n") 
            print(output)

            analysis = match.group(1).strip()
            subtask = match.group(2).strip()
            model_choice = match.group(3).strip()
            homie.comm.send_subtask(f"{subtask}|{model_choice}|{homie.get_inventory()}")

            feedback, signal = homie.comm.receive_feedback()
            homie.update_inventory(subtask, feedback)
            historical_execution += f"({homie.conv.round}) {subtask}({signal}) "
            print(historical_execution)

            if "end" in subtask.lower():
                break

        homie.conv.save(os.path.join(save_path, "conversation.json"))

homie.end()