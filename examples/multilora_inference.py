"""
This example shows how to use the multi-LoRA functionality
for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import List, Optional, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from vllm.stream_pool_manager import StreamPoolManager

import torch
import time

def create_test_prompts(
        lora_path: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
        # ("A robot may not injure a human being",
        #  SamplingParams(temperature=0.0,
        #                 logprobs=1,
        #                 prompt_logprobs=1,
        #                 max_tokens=128), None),
        # ("To be or not to be,",
        #  SamplingParams(temperature=0.8,
        #                 top_k=5,
        #                 presence_penalty=0.2,
        #                 max_tokens=128), None),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            #None
            LoRARequest("sql-lora", 1, lora_path)
        ),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           #prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            #None
            LoRARequest("sql-lora2", 2, lora_path)
        ),
    ]
    
def create_dummy_test_prompts(
    number_of_requests : int,
    prompt_len : int,
    lora_path: str,
    using_lora_num: Optional[int],
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    requests = []
    
    embedded_prompt_len = prompt_len - 1
    prompt = "Hi" + (embedded_prompt_len - 1) * " Hi"
    #prompt = "The quick brown fox"
    sample_parms = SamplingParams(temperature=0.0,
                           logprobs=1,
                           #prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003])
    
    if lora_path == "":
        using_loras = [None] * number_of_requests
    else:
        lora_per_used = (number_of_requests / using_lora_num)
        
        using_lora_ids = [int(i / lora_per_used) for i in range(number_of_requests)]
        using_loras = [LoRARequest(f"sql-lora {i+1}", i+1, lora_path) for i in using_lora_ids]
        
    for i in range(number_of_requests):
        request = (prompt, sample_parms, using_loras[i])
        requests.append(request)
        
    return requests


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    loop = 0
    # while test_prompts or engine.has_unfinished_requests():
    while test_prompts:
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

    step = 0
    while engine.has_unfinished_requests():
        torch.cuda.nvtx.range_push("Step")
        request_outputs: List[RequestOutput] = engine.step()
        torch.cuda.nvtx.range_pop()
        if request_outputs == None:
            return
        #loop += 1
        #print(loop)
        step += 1
        print(f"Step {step}")
        if step > 10:
            print(f"Exited :: ======================================")
            break
        
        for i, request_output in enumerate(request_outputs):
            if request_output.finished:
                print(f"Finished {i}:: ======================================")
                print(request_output)
        

def initialize_engine(batch_size : int, prompt_len : int) -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    enable_lora=True
    # if enable_lora == True:
    #     StreamPoolManager.instance()
    engine_args = EngineArgs(model="meta-llama/Llama-2-7b-hf",
                             #model="meta-llama/Llama-3.1-8B",
                             enable_lora=enable_lora,
                             use_punica=True,
                             max_loras=32,
                             max_lora_rank=8,
                             max_cpu_loras=64,
                             max_num_seqs=batch_size,
                             max_model_len=prompt_len + 10,
                             max_num_batched_tokens=batch_size * (prompt_len + 10),
                             gpu_memory_utilization=0.65,
                             enable_chunked_prefill=False,
                             #enforce_eager=True
    )
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    batch_size = 4
    prompt_len = 128

    #torch.cuda.nvtx.range_push("Initializing engine")
    engine = initialize_engine(batch_size, prompt_len)
    #torch.cuda.nvtx.range_pop()
    # print("\nSleeping...")
    # time.sleep(60)
    
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    #lora_path = snapshot_download(repo_id="RikiyaT/Meta-Llama-3.1-8B-LoRA-test")
    #test_prompts = create_test_prompts(lora_path)
    

    #test_prompts = create_dummy_test_prompts(batch_size, prompt_len, "")
    test_prompts = create_dummy_test_prompts(batch_size, prompt_len, lora_path, batch_size)
    
    process_requests(engine, test_prompts)
    print(batch_size)

if __name__ == '__main__':
    main()
