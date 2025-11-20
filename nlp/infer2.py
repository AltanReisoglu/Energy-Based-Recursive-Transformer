import torch
import argparse
from types import SimpleNamespace
from ebt import EBT_NLP
from turkish_tokenizer import HFTurkishTokenizer
import os
from dotenv import load_dotenv
load_dotenv()
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def default_hparams():
    """Default hyperparameters matching train_ebt.py"""
    hparams = dict(
        # optimisation
        lr=1e-3,
        batch_size_per_device=2,
        num_workers_per_gpu=12,
        max_steps=100000,

        # data
        dataset_dir="",
        dataset_name="selimfirat/bilkent-turkish-writings-dataset",
        context_length=256,
        pretokenize_dataset=True,
        

        # model choice
        model_name="ebt",  # "baseline_transformer" or "ebt"

        # model size
        embedding_dim=256,
        num_transformer_blocks=6,
        multiheaded_attention_heads=6,
        ffn_dim_multiplier=4,
        weight_initialization_method="xavier",
        weight_initialization_gain=1.0,
        model_max_length=256,
        # misc
        execution_mode="inference",
        debug_unused_parameters=False,
        mcmc_step_size=500.0,
        num_workers=4,
        num_gpus=1
    )

    ebt_params = dict(
        model_max_length=64,
        mcmc_step_size=500.0,
        model_name="ebt",
        mcmc_step_size_lr_multiplier=1500.0,
        mcmc_num_steps=3,
        ebt_type="default",
        normalize_initial_condition=True,
        denoising_initial_condition="random_noise",
        mcmc_step_size_learnable=True,
        no_mcmc_detach=False,
        ebt_norm="rms",
        ebt_act_func="silu",
        dyt_alpha_init=0.5,
        mcmc_replay_buffer=False,
        gaussian_random_noise_scaling=1.0,
        normalize_initial_condition_only_first_step=False,
        randomize_mcmc_step_size_scale=1.0,
        randomize_mcmc_num_steps=0,
        randomize_mcmc_num_steps_min=0,
        randomize_mcmc_num_steps_final_landscape=False,
        langevin_dynamics_noise=0.0,
        langevin_dynamics_noise_learnable=False,
        vocab_to_embed_uses_prob_dist=False,
        num_modality_processing_mlp_layers=1,
        truncate_mcmc=False,
        clamp_futures_grad=False,
        clamp_futures_grad_max_change=9.0,
        absolute_clamp=0.0,
        clamp_max_after_warm_up=0.0,
        sharpen_predicted_distribution=0.0,
        reconstruction_coeff=1.0,
        contrastive_loss=False,
        contrastive_loss_coeff=0.0005,
        soften_target_prob_dist=0.0,
        adaptive=False,
        infer_ebt_override_alpha=0.,
        infer_generated_samples=1,
        infer_steps_final_landscape=False,
        infer_energy_sampling_technique="min",
        infer_alpha_final_landscape=False,
        infer_langevin_first_step=True,
        infer_accept_lower_energies=True,
        infer_ebt_num_steps=1,
        gradient_accumulation_steps=4,
        use_amp=False,
        use_activation_checkpointing=True,
        use_torch_compile=True,
        use_bnb_optimizer=False,
        infer_langevin_dynamics_noise=0
    )
    hparams.update(ebt_params)

    hparams = SimpleNamespace(**hparams)
    return hparams
    


def load_checkpoint_to_model(model, ckpt_path, device):
    """Load checkpoint into model, handling both state_dict and full ckpt formats"""
   
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    
    try:
        model.load_state_dict(state, strict=True)
        print("✓ Checkpoint loaded (strict)")
    except RuntimeError as e:
        print(f"⚠ Strict load failed, trying non-strict: {e}")
        model.load_state_dict(state, strict=False)
        print("✓ Checkpoint loaded (non-strict)")
    
    return model


def prepare_inputs(prompts, tokenizer, max_len, device):
    """Tokenize prompts and prepare input tensors"""
    enc = tokenizer(prompts, padding='max_length', truncation=True, 
                    max_length=max_len, return_tensors='pt')
    input_ids = enc['input_ids']
    return input_ids.to(device)


def decode_tokens(tokenizer, token_ids):
    """Decode token IDs back to text"""
    outs = []
    for ids in token_ids.cpu().numpy().tolist():
        try:
            txt = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            toks = [tokenizer.convert_ids_to_tokens(i) for i in ids]
            txt = ''.join(toks)
        outs.append(txt)
    return outs


def _move_obj_to_device(obj, device):
    """Recursively move tensors in nested object to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_obj_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_obj_to_device(item, device) for item in obj)
    elif hasattr(obj, '__dict__'):
        new_obj = object.__new__(type(obj))
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, _move_obj_to_device(v, device))
        return new_obj
    else:
        return obj


def generate(
    model,
    input_ids,
    tokenizer,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    device='cpu',
    eos_token_id=None
):
    """
    Generate tokens using cache and carry, stopping at EOS token.
    
    Args:
        model: EBT_NLP model
        input_ids: (batch_size, prompt_len) tensor
        tokenizer: HFTurkishTokenizer instance
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        device: torch device
        eos_token_id: token ID to stop generation (default: tokenizer.eos_token_id)
    
    Returns:
        generated_ids: (batch_size, prompt_len + num_generated) tensor
    """
    batch_size, prompt_len = input_ids.shape
    
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    # Initialize carry and cache
    model.eval()
    with torch.no_grad():
        carry = model.initial_carry(batch_size,2 * prompt_len )
        carry = _move_obj_to_device(carry, device)
        past_cache = None
        
        # Process prompt
        """print(carry.inner_carry.z_H.shape,"1111111")
        print(input_ids.shape,"11111111111111")"""
        logits, energies, pred_dists, past_cache, carry = model.ebt_advanced_inference(
            input_ids, carry=carry, past_cache=past_cache
        )
        
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for step in range(5):
            # Get last logits (batch_size, vocab_size)
            next_logits = logits[:, -1,:]  # (batch_size, vocab_size)
            
            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            
            # Get probabilities
            next_probs = torch.softmax(next_logits, dim=-1)
            
            # Sample or greedy
            if top_p < 1.0:
                # Top-p sampling per example
                next_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                for i in range(batch_size):
                    if not finished[i]:
                        next_token[i] = sample_top_p(next_probs[i:i+1], top_p)
                    else:
                        next_token[i] = eos_token_id
            else:
                # Greedy (argmax)
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            for i in range(batch_size):
                if next_token[i, 0].item() == eos_token_id:
                    finished[i] = True
            
            # Stop if all finished
            if finished.all():
                break
            
            # Forward pass for next step (only for unfinished examples)
            logits, energies, pred_dists, past_cache, carry = model.ebt_advanced_inference(
                generated[:,-prompt_len:], carry=carry, past_cache=None
            )

            
    
    return generated


def main():
    args=SimpleNamespace(
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        prompt="Belli etmemeye çalışsak da hepimiz ölümden",
        ckpt =os.getenv(key="ckpt"),
        batch_size=1,
        max_len=128,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=128
    )
    
    # Determine device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    hparams = default_hparams()
    config_dict = dict(H_cycles=1, L_cycles=1, H_layers=1, L_layers=1)
    model = EBT_NLP(hparams,config_dict).to(device)
    load_checkpoint_to_model(model, args.ckpt, device)
    
    tokenizer = HFTurkishTokenizer(
            bos_token="<s>",
            eos_token="</s>",
            sep_token="<sep>",
            cls_token="<cls>",
            mask_token="<mask>",
            pad_token="<pad>",
            unk_token="<unk>",
            model_max_length=128
        )
    
    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    
    
    # Process in batches
    results = []
    for batch_idx in range(0, len(prompts)):
        
        

        input_ids = prepare_inputs(prompts, tokenizer, args.max_len, device).unsqueeze(0)
        input_ids=torch.cat((input_ids,input_ids),dim=0)
        print(input_ids.shape)
        # Generate
        generated_ids = generate(
            model, input_ids, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode and print
        texts = decode_tokens(tokenizer, generated_ids)
        for prompt, text in zip(prompts, texts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {text}\n")
            results.append({'prompt': prompt, 'generated': text})
    

        """    if args.out_file:
        import json
        with open(args.out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.out_file}")"""


if __name__ == '__main__':
    main()