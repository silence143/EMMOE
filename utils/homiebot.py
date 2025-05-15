from .conv import Conversation
from .comm import Communicator_HLP
import os

class HomieBot:
    def __init__(self, port=10010):
        self.conv = Conversation()
        self.inventory = []
        self.comm = Communicator_HLP(port)

    def get_inventory(self):
        if len(self.inventory) == 0:
            return "None"
        else:
            return " ".join(self.inventory)

    def generate_instruction(self, task, feedback, historical_execution):
        if historical_execution == "":        
            instruction = f"Task: {task}\nInventory: {self.get_inventory()}\nHistorical Execution: None\nFeedback: None\nNow based on the instruction above, please output Analysis, Subtask and Model in the mentioned format.\n"    
        else:     
            instruction = f"Task: {task}\nInventory: {self.get_inventory()}\nHistorical Execution: {historical_execution}\nFeedback: {feedback}\nNow based on the instruction above, please output Analysis, Subtask and Model in the mentioned format.\n"
        return instruction

    def update_inventory(self, subtask, feedback):
        subtask = subtask.lower()
        if "None" in feedback:
            if "pick" in subtask:
                obj = subtask.split(']')[0].split(',')[1].strip()
                self.inventory.append(obj)
            if "put" in subtask:
                self.inventory.pop()     
        else:
            if "put" in subtask and "the object is missing" in feedback:
                self.inventory.pop()
    
    def end(self, save_dir=None):
        self.comm.close_connection()
        if save_dir != None:
            self.conv.save(os.path.join(save_dir, "conv.json"))
