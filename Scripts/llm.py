import torch
import requests
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig



import preprocess
import numpy as np
import faiss
import textwrap
import constants


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
    return scores[0], indices[0]



def get_llm_selection(constants):
    """
    Selects an appropriate LLM model based on the available GPU memory.
    """
    device = "gpu" if torch.cuda.is_available() else "cpu"
    if device == "gpu":
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
    constants["DEVICE"].update(device)
    constants["LLM"]["use_quantization"].update(use_quantization_config)
    constants["LLM"]["model_name"].update(model_id)

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
    base_prompt = """Based on the following context items, please answer the query.
                    Give yourself room to think by extracting relevant passages from the context before answering the query.
                    Don't return the thinking, only return the answer.
                    Make sure your answers are as explanatory as possible.
                    Use the following examples as reference for the ideal answer style.
                    \nExample 1:
                    Query: What are the fat-soluble vitamins?
                    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
                    \nExample 2:
                    Query: What are the causes of type 2 diabetes?
                    Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
                    \nExample 3:
                    Query: What is the importance of hydration for physical performance?
                    Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
                    \nNow use the following context items to answer the user query:
                    {context_items}
                    \nRelevant passages: <extract relevant passages from the context here>
                    User query: {query}
                    Answer:"""

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context_items, query=query)
    return base_prompt

    

def ask_llm(query,
            metadata,
            tokenizer,
            model,
            temperature=constants["LLM"]["temperature"],
            max_new_tokens=constants["LLM"]["max_new_tokens"],
            format_answer_text=True, 
            return_answer_only=True,
            device = constants["DEVICE"]):
    """
    Asks the LLM model a question augmented with context retrieved from the metadata and returns the response.
    """
    scores, indices = get_similarity_search(query=query)
    context_items =  {}     # Make it a List of dict with scores 
    for index in indices[0]:
        context_items["context"].append(f"Category: {metadata.iloc[index]['category']}, Text: {metadata.iloc[index]['sentence_chunk']}")
    
    # Add score to context item
    for score in scores:
        context_items["score"].append(score.cpu()) # return score back to CPU 

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
        print(f"Context items: {context_items}")

    