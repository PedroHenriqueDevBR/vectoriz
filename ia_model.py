import requests

class IaModel:
    
    def __init__(self):
       self.url = "http://localhost:11434/api/generate"
       self.model = "gemma3:4b"
    
    def generate(self, prompt):
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            
        }
        response = requests.post(self.url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
