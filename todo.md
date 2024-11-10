# TODO

- Use hf.co/bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF:Q3_K_XL  to evaluate the following candidate models:
    - hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_L                        ebf52558c6bb    5.8 GB    10 hours ago    
    - hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q5_K_L                 47e8fc97e160    6.1 GB    21 hours ago    
    - hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:F16                         68bfdddd17b0    6.4 GB    21 hours ago    
    - hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:F16                         108b021e89a8    2.5 GB    21 hours ago    
    - hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:Q2_K                         cda26c1761a3    5.8 GB    21 hours ago    
    - hf.co/bartowski/Qwen2.5-3B-Instruct-GGUF:F16                           43aa152f1029    6.2 GB    21 hours ago    
    - hf.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF:F16                         a64bfa068379    3.1 GB    21 hours ago    
    - hf.co/bartowski/Phi-3-medium-128k-instruct-GGUF:Q3_K_S                 74f8ad8b9698    6.1 GB    21 hours ago    
    - hf.co/RichardErkhov/princeton-nlp_-_gemma-2-9b-it-SimPO-gguf:Q4_K_M    016add2b142a    5.8 GB    22 hours ago 

- Important Tasks
    - Base Training
        - **Forgot previous instructions though even the original model does this**
        - Repetition
        - Not stopping
        - Hallucination
        - Incoherence
        
    - Instruction Tuning
    - Dataset
        - Augmentoolkit
            - Use bigger model
            - Try different prompt engineering techniques
        - Semantic Chunking vs Chapter-wise
        - Current implementation sliding window with minimal overlap
    - Prompt Engineering
        - dspy
        - textgrad
    - RAG
        - Microsoft GraphRAG
        - LightRAG
        - Latest in GraphRAG

- Inference
    - Research on extremely fast inference techniques
        - llamafile
        - SGLang for long context caching with vLLM


- Future Upgrades
    - Distillate Nemotron Model to fit in 8GB VRAM Macbook Air M1 2020
    - Router LLMs for creating better prompts before being fed into the main model


================================================================================================================================================================================================================================================================================================
- Inspiration for other projects
    - Dataset Curation
    - STT
    - TTS
    - Image Generation
    - Code Generation
    - Chatbot
