import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig



import preprocess
import numpy as np
import faiss
import textwrap
import json


def load_constants(json_path='constants.json'):
    with open(json_path, 'r') as file:
        return json.load(file)

# Load constants at the start
constants = load_constants()

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def get_similarity_search(query, 
                          embed_dir = constants["SAVED_DIR"], 
                          model_name = constants["EMBED_MODEL"], 
                          device_name = constants["DEVICE"],
                          num_nearest_search = constants["PARAMS"]["num_nearest_search"]):
    """
    Searches for nearest neighbors in pre-processed embeddings for a given query.
    """

    embeddings, metadata = preprocess.load_embeddings_and_metadata(embed_dir)
    embedding_model = preprocess.get_embedding_model(model_name, device_name)
    query_embedding = np.array(embedding_model.encode(query))
    
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # faiss search
    embed_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embed_dimension)
    index.add(embeddings)
    scores, indices = index.search(query_embedding, num_nearest_search)

    context_items =  {
                        "context": [],
                        "score" : []
                      }      
    for index, score in zip(indices[0], scores[0]):
        text = f"Title: {metadata.iloc[index]['title']}\n Category: {metadata.iloc[index]['category']}\n Text: {metadata.iloc[index]['sentence_chunk']}\n"
        print_wrapped(text)
        print(200*"*")
        context_items["context"].append(text)
        context_items["score"].append(score)

    return context_items



def get_llm_selection(constants):
    """
    Selects an appropriate LLM model based on the available GPU memory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_memory_bytes / (2**30))
        print(f"[INFO] Available GPU memory: {gpu_memory_gb} GB")
        if gpu_memory_gb < 5.1:
            print(f"[INFO] Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
        elif gpu_memory_gb < 8.1:
            print(f"[INFO] GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
            use_quantization_config = True
            model_id = "google/gemma-2b-it"
        elif gpu_memory_gb < 19.0:
            print(f"[INFO] GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
            use_quantization_config = False
            model_id = "google/gemma-2b-it"
        elif gpu_memory_gb > 19.0:
            print(f"[INFO] GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
            use_quantization_config = False
            model_id = "google/gemma-7b-it"
    else:
        print(f"[INFO] No GPU memory found on the device. | Recommended model: Gemma 2B with no precision support.")
        use_quantization_config = False

    # Update the settings on json file
    # Update the settings on json file
    constants["DEVICE"] = device
    constants["LLM"]["use_quantization"] = use_quantization_config
    constants["LLM"]["model_name"] = model_id

    with open("./constants.json", "w") as file:
      json.dump(constants, file, indent=4)

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")



def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes
    print(f"model_mem_gb: {round(model_mem_gb, 2)}")



def get_llm_model(model_name, use_quantization_config, device, set_flash_attention = True):
    """
    Loads a language model with optional quantization and flash attention settings.
    """
    # Set Flash Attention for faster Inference
    if set_flash_attention and (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
        print(f"[INFO] Using attention implementation: {attn_implementation}")

    # Use quantization with the chosen precision
    if use_quantization_config:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)
    else:
        quantization_config = None
        print(f"[INFO] Quantization is turned off.")
        
    # Instantiate Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                    torch_dtype=torch.float16, 
                                                    low_cpu_mem_usage=False,
                                                    attn_implementation=attn_implementation,
                                                    quantization_config=quantization_config).to(device)
    # Get model params
    get_model_mem_size(llm_model), print(llm_model)
    return tokenizer, llm_model



def prompt_formatter(query, context_items):
    """
    Augments query with text-based context from context_items.
    """

    # Create a base prompt with examples to help the model
    base_prompt = """Use the following context items to answer the user query:
                    {context}
                    \nRelevant passages: <extract relevant passages from the context here>
                    User query: {query}
                    Answer:"""

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context_items, query=query)
    return base_prompt


def ask_llm(query,
            tokenizer,
            model,
            temperature,
            max_new_tokens,
            device, 
            return_answer_only=False,
            format_answer_text=True):
    """
    Asks the LLM model a question augmented with context retrieved from the metadata and returns the response.
    """
    context_items = get_similarity_search(query=query)
    

    # Format the prompt with context items
    base_prompt = prompt_formatter(query=query,
                              context_items=context_items)
    
    # Create prompt template for instruction-tuned model
    dialogue_template = [
                            {
                                "role": "user",
                                 "content": base_prompt
                            }
                        ]
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate an output of tokens
    outputs = model.generate(**input_ids,
                                temperature=temperature,
                                do_sample=True,
                                max_new_tokens=max_new_tokens
                            )
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])
    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")
    
    if return_answer_only:
        print(f"Answer:\n")
        print_wrapped(output_text)
    else:
        print(f"Answer:\n")
        print_wrapped(output_text)
        print(f"Context:\n")
        print_wrapped(context_items)
    