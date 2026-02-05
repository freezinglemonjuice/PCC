import torch
import accelerate

@torch.no_grad()
def test_latency(model, generation) :

    if (generation) :
        generation_length = 128
        prompt_length = 64
        batch_size = 64
        max_length = prompt_length + generation_length
        model.config.max_length = max_length
        model.config.use_cache = True
        model.generation_config.use_cache = True
        iteration = 10

        random_input = torch.randint(0, 31999, (batch_size, prompt_length), dtype=torch.long)
        random_input = random_input.to(model.device).contiguous()

        model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)

        starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        starter.record()
        for i in range(iteration):
            model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)
        ender.record()
        torch.cuda.synchronize()

    else :
        batch_size = 1
        model.config.use_cache = False
        model.generation_config.use_cache = False
        iteration = 50

        random_input = torch.randint(0, 31999, (batch_size, 2048), dtype=torch.long)
        random_input = random_input.to(model.device).contiguous()
        
        model(random_input)

        starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        starter.record()
        for i in range(iteration):
            model(random_input)
        ender.record()
        torch.cuda.synchronize()

    curr_time = starter.elapsed_time(ender)
    mean_latency = curr_time/iteration

    return mean_latency
