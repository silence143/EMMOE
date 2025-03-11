import re   
import os
import json
import socket
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from utils import HomieBot


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

exp_dir = "exp"
exp_name = "MiniCPM"
exp_path = os.path.join(exp_dir, exp_name)

model = AutoModel.from_pretrained('checkpoints/MiniCPM-V-2_6', 
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('checkpoints/MiniCPM-V-2_6', trust_remote_code=True)

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
            image_loads = [Image.open(image).convert('RGB') for image in images]

            match = None
            for i in range(5):
                messages = [
                    {"role": "user",
                     "content": "here is an example output, please strictly follow its format and system reminders in your output:\nAnalysis: According to my final task, I need to fetch apples first, but it's a better choice to go the fridge and open it first, which will avoid potential conflicts, so I should go to the fridge next\nSubtask: [Go to, fridge]\nModel: NoMaD\n",
                    },
                    {"role": "assistant",
                     "content": "I will surely follow the given format, now you can send prompt to me.",
                    },
                    {'role': 'user', 'content': [image_loads[0], image_loads[1], image_loads[2], image_loads[3], instruction]}]
                
                output = model.chat(
                    image=None,
                    system_prompt=homie.conv.system,
                    msgs=messages,
                    tokenizer=tokenizer
                )
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