import re   
import os
import json
import socket
import base64
from openai import OpenAI
from utils import HomieBot


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

exp_dir = "exp"
exp_name = "GPT4"
exp_path = os.path.join(exp_dir, exp_name)


client = OpenAI("YOUR API HERE")
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
            image_encodes = []
            for image_path in images:
                image_encodes.append(encode_image(image_path))
            
            match = None
            for i in range(5):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": homie.conv.system},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_encodes[0]}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_encodes[1]}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_encodes[2]}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_encodes[3]}"}},
                            ],
                        },
                    ],
                    max_tokens=300,
                )  
                output = response.choices[0].message.content.strip()
                    
                pattern = r'.*Analysis: *(.+?) *Subtask: *\[(.*?)\].*Model: *(.*?)$'
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    break

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