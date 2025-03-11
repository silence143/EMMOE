import torch
import pickle
import os
import json
from PIL import Image
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

class RAG:
    def __init__(self, buffer_path, subtask_list, save_path="RAG"):
        self.text_tokenizer = AutoTokenizer.from_pretrained("local_model/bge-small-en-v1.5")
        self.text_model = AutoModel.from_pretrained("local_model/bge-small-en-v1.5")
        self.image_model = CLIPModel.from_pretrained("local_model/clip-vit-large-patch14")
        self.image_processor = CLIPProcessor.from_pretrained("local_model/clip-vit-large-patch14")
        self.success_buffer = []
        self.failure_buffer = []
        if buffer_path:
            self.load_buffer(buffer_path)
        else: 
            self.build_buffer(subtask_list, save_path)

    def create_node(self, instruction, image_paths):        
        new_node = {}
        new_node["instruction"] = instruction
        new_node["images"] = image_paths

        self.text_model.eval()
        encoded_ins = self.text_tokenizer(instruction, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.text_model(**encoded_ins)
            ins_embedding = model_output[0][:, 0]
        new_node["text_embeddings"] = torch.nn.functional.normalize(ins_embedding, p=2, dim=1)

        image_embeddings = []
        for image_path in image_paths:
            img_inputs = self.image_processor(images=Image.open(image_path), return_tensors="pt", padding=True)
            with torch.no_grad():
                image_embedding = self.image_model.get_image_features(**img_inputs)
            image_embeddings.append(torch.nn.functional.normalize(image_embedding, p=2, dim=1))
            
        new_node["image_embeddings"] = image_embeddings
        return new_node
    
    def build_buffer(self, subtask_list, save_path):
        flag = 1
        for subtask in subtask_list:
            instruction = subtask['conversations'][0]['value']
            new_node = self.create_node(instruction, subtask['image'])
            answer = subtask['conversations'][1]['value']
            if "Feedback: None" in instruction:
                if "End" in answer:
                    new_node["analysis"] = answer.split('\n')[0]
                    new_node["subtask"] = "End"
                    new_node["model_choice"] = "None"
                else:
                    analysis, subtask, model_choice = answer.split('\n')
                    new_node["analysis"] = analysis[10:]
                    new_node["subtask"] = subtask[9:]
                    new_node["model_choice"] = model_choice[7:]            

                if flag == 0:
                    self.failure_buffer.append(new_node)
                    flag = 1
                else:
                    self.success_buffer.append(new_node)
            else: 
                flag = 0

        print(f"success buffer: {len(self.success_buffer)}")
        print(f"failure buffer{len(self.failure_buffer)}")

        with open(os.path.join(save_path, 'success.pkl'), 'wb') as f:
            pickle.dump(self.success_buffer, f)

        with open(os.path.join(save_path, 'failure.pkl'), 'wb') as f:
            pickle.dump(self.failure_buffer, f)

    def load_buffer(self, buffer_path):
        with open(os.path.join(buffer_path, 'success.pkl'), 'rb') as f:
            self.success_buffer = pickle.load(f)
        with open(os.path.join(buffer_path, 'failure.pkl'), 'rb') as f:
            self.failure_buffer = pickle.load(f)

    def search(self, subtask_node, is_replan, weight=0.8):
        from torch.nn.functional import cosine_similarity
            
        buffer = self.failure_buffer if is_replan else self.success_buffer
        scores = [[idx, cosine_similarity(subtask_node["text_embeddings"], node["text_embeddings"]).item()] for idx, node in enumerate(buffer)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
        for score in scores:
            idx = score[0]
            score[1] = score[1] * weight
            for query_emb, buffer_emb in zip(subtask_node["image_embeddings"], buffer[idx]["image_embeddings"]):
                score[1] += cosine_similarity(query_emb, buffer_emb).item() * (1 - weight) / 4
                
        target = max(scores, key=lambda x: x[1])

        return target[1], buffer[target[0]]['subtask'], buffer[target[0]]['model_choice']

    def update(self, success_nodelist, failure_nodelist):
        self.success_buffer.extend(success_nodelist)
        self.failure_buffer.extend(failure_nodelist)
            
    def save(self, save_dir):
        with open(os.path.join(save_dir, 'success.json'), 'w') as file:
            json.dump(self.success_buffer, file, indent=4)
        with open(os.path.join(save_dir, 'failure.json'), 'w') as file:
            json.dump(self.failure_buffer, file, indent=4)