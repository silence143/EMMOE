import torch
import numpy as np
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
import os
from PIL import Image
from utils import HomieBot

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
disable_torch_init()


model_path = 'checkpoints/HomieBot-7B-DPO'
#model_path = 'checkpoints/HomieBot-7B-SFT'

cache_dir = 'cache_dir'
device = 'cuda:0'
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_4bit=True, device=device, cache_dir=cache_dir)
image_processor = processor['image']

task = input("Input your task: ")
homie = HomieBot()

for i in range(1, 4):
    save_path = f"infer/{i}"
    homie.conv.reset()
    homie.inventory = []
    feedback = ""
    historical_execution = ""

    while homie.conv.round < homie.conv.max_round:
        homie.conv.round += 1
        instruction = homie.generate_instruction(task, feedback, historical_execution)
        images = homie.comm.receive_env_images()
        print(images)
            
        prompt = homie.conv.system + " USER: " +  DEFAULT_IMAGE_TOKEN + '\n' + instruction + " ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        if type(image_tensor) is list:
            tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            tensor = image_tensor.to(model.device, dtype=torch.float16)
                
        keywords = [homie.conv.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        match = None
        for i in range(5):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])  
            output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
 
            import re
            pattern = r'.*Analysis: *(.+?) *Subtask: *\[(.*?)\].*Model: *(.*?)$'
            match = re.search(pattern, output, re.DOTALL)
            if match:
                break
            pattern = r'.*Analysis: *(.+?) *Subtask: *(.*?) *Model: *(.*?)$'
            match = re.search(pattern, output, re.DOTALL)
            if match:
                break
            # print(output)    

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
print("All data are saved in ./infer")