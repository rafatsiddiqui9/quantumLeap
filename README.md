# Instruction Dataset creation

##  Augmentoolkit config file for quantumLeap
```
git clone https://github.com/e-p-armstrong/augmentoolkit
cd augmentoolkit
uv pip install -r requirements.txt
```


# In another terminal run the following command to start the API server
```
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 32768
```

### To check if the server is running
```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
```








