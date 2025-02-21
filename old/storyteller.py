import os
import copy
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.logging import RichHandler
import logging

# ---------------------
# Setup Rich Logging
# ---------------------
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("rich")

# ---------------------
# Hyperparameters & Defaults
# ---------------------
GROUP_SIZE = 3               # Number of candidate stories per prompt
MAX_GEN_LEN = 200            # Maximum new tokens to generate per candidate
EPSILON = 0.2                # Clipping parameter for GRPO
KL_COEFF = 0.01              # Coefficient for KL penalty
LR = 1e-5                    # Learning rate
UPDATE_INTERVAL = 2          # Update reference model every N training steps
NUM_RANKING_PASSES = 3       # Number of passes for the model-based ranking evaluator
TOTAL_TRAIN_STEPS = 1000     # Total number of training steps

# Reward weights
ALPHA_MODEL = 1.0            # Weight for model-based reward (narrative quality)
BETA_RULE = 1.0              # Weight for rule-based (length) reward
TARGET_WORD_COUNT = 100      # Target word count for a "short" story

# ---------------------
# Rule-Based Reward: Encourages short stories
# ---------------------
def compute_rule_reward(story: str, target_length: int = TARGET_WORD_COUNT) -> float:
    word_count = len(story.split())
    reward = max(0.0, (target_length - word_count) / target_length)
    return reward

# ---------------------
# Model-Based Reward Evaluator for Narrative Quality (with emphasis on twists)
# ---------------------
def self_critic_rank_story(prompt: str, stories: list, tokenizer, model, device, num_passes: int = NUM_RANKING_PASSES) -> list:
    num_candidates = len(stories)
    aggregated_scores = np.zeros(num_candidates, dtype=np.float32)
    
    for _ in range(num_passes):
        perm = np.random.permutation(num_candidates)
        shuffled_stories = [stories[i] for i in perm]

        evaluator_prompt = (
            "You are a narrative critic with a keen eye for plot twists and suspense. "
            "The current prompt is " + prompt + ", so critic intensely based on adherence to the prompt"
            "Evaluate the following short stories based on overall brevity, and suspense,"
            "the presence of surprising twists and suspenseful moments. "
            "Provide ONLY a comma-separated list of numbers ranking these stories from best (highest quality) to worst. For example: 2,1,3\n\n"
            "Stories to rank:\n"
        )
        for idx, story in enumerate(shuffled_stories):
            evaluator_prompt += f"{idx+1}. {story}\n"
        
        with torch.no_grad():
            inputs = tokenizer(evaluator_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            # 'model' is DDP wrapped so we use model.module.generate
            ranking_output = model.module.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        ranking_text = tokenizer.decode(ranking_output[0].cpu(), skip_special_tokens=True).splitlines()[0].strip()

        try:
            ranking_numbers = [int(x.strip()) for x in ranking_text.split(",") if x.strip().isdigit()]
            if len(ranking_numbers) != num_candidates:
                ranking_numbers = list(range(1, num_candidates + 1))
        except Exception as e:
            logger.warning(f"Ranking parse error: {e}. Using default ranking.")
            ranking_numbers = list(range(1, num_candidates + 1))
        
        scores = np.zeros(num_candidates, dtype=np.float32)
        for rank_pos, candidate_label in enumerate(ranking_numbers):
            original_idx = perm[candidate_label - 1]
            scores[original_idx] = num_candidates - rank_pos  # Best gets highest score
        
        aggregated_scores += scores

    final_model_rewards = aggregated_scores / num_passes
    return final_model_rewards.tolist()

# ---------------------
# Log Probability Function for GRPO
# ---------------------
def compute_log_prob(model, input_ids, gen_ids):
    full_ids = torch.cat([input_ids, gen_ids], dim=-1)
    outputs = model(full_ids)
    logits = outputs.logits
    L_prompt = input_ids.shape[-1]
    gen_logits = logits[:, L_prompt - 1:-1, :]
    log_probs = F.log_softmax(gen_logits, dim=-1)
    gen_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
    total_log_prob = gen_log_probs.sum(dim=-1)
    return total_log_prob

# ---------------------
# Prompt Augmentation for Story Generation
# ---------------------
def augment_prompt(prompt: str) -> str:
    augmented = (
        "You are a creative short story writer. Based on the prompt below, generate an engaging, concise short story. "
        "Your story should be a single coherent narrative that surprises the reader with its creativity.\n\n"
        "Prompt:\n" + prompt + "\n\nStory:"
    )
    return augmented

# ---------------------
# Self-Generated Creative Writing Prompt Function
# ---------------------
def generate_creative_prompt(tokenizer, model, device) -> str:
    """
    Generate a creative writing prompt using a gradient-free forward pass with the same model.
    A new random seed is set on each call for diversity.
    The instruction ensures that the model returns ONLY the prompt text.
    """
    base_instruction = "Generate a unique and interesting creative writing prompt. Return ONLY the prompt text."
    with torch.no_grad():
        inputs = tokenizer(base_instruction, return_tensors="pt", truncation=True, max_length=64).to(device)
        new_seed = random.randint(0, 1000000)
        torch.manual_seed(new_seed)
        # Since 'model' here is not wrapped in DDP (old_model), call generate() directly.
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_prompt = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return generated_prompt

# ---------------------
# Training Step Function
# ---------------------
def training_step(prompt: str, tokenizer, model, old_model, optimizer, device, training_log, global_step):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)

    candidate_gen_ids = []
    candidate_stories = []
    for _ in range(GROUP_SIZE):
        gen_output = old_model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + MAX_GEN_LEN,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        gen_ids = gen_output[:, input_ids.shape[-1]:].to(device)
        candidate_gen_ids.append(gen_ids)

        full_text = tokenizer.decode(gen_ids[0].cpu(), skip_special_tokens=True)
        story = full_text.split("Story:")[-1].strip()
        candidate_stories.append(story)

    torch.cuda.empty_cache()

    # Compute rewards
    rule_rewards = [compute_rule_reward(story) for story in candidate_stories]
    model_rewards = self_critic_rank_story(prompt, candidate_stories, tokenizer, model, device)
    combined_rewards = [ALPHA_MODEL * m + BETA_RULE * r for m, r in zip(model_rewards, rule_rewards)]
    
    rewards_np = np.array(combined_rewards)
    mean_reward = rewards_np.mean()
    std_reward = rewards_np.std() + 1e-8
    normalized_advantages = [(r - mean_reward) / std_reward for r in combined_rewards]

    worst_idx = int(np.argmin(combined_rewards))
    filtered_advantages = [normalized_advantages[i] for i in range(GROUP_SIZE) if i != worst_idx]
    filtered_gen_ids = [candidate_gen_ids[i] for i in range(GROUP_SIZE) if i != worst_idx]

    candidate_objs = []
    kl_losses = []
    for gen_ids, adv in zip(filtered_gen_ids, filtered_advantages):
        logp_new = compute_log_prob(model, input_ids, gen_ids)
        with torch.no_grad():
            logp_old = compute_log_prob(old_model, input_ids, gen_ids)
        ratio = torch.exp(logp_new - logp_old)
        clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
        adv_tensor = torch.tensor(adv, device=device)
        candidate_obj = torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor) if adv_tensor >= 0 \
                         else torch.max(ratio * adv_tensor, clipped_ratio * adv_tensor)
        candidate_objs.append(candidate_obj)

        with torch.no_grad():
            outputs_old = old_model(input_ids)
            outputs_new = model(input_ids)
        log_probs_old = F.log_softmax(outputs_old.logits, dim=-1)
        log_probs_new = F.log_softmax(outputs_new.logits, dim=-1)
        kl_div = F.kl_div(log_probs_new, log_probs_old, reduction="batchmean", log_target=True)
        kl_losses.append(kl_div)

    surrogate_loss = -torch.stack(candidate_objs).mean()
    loss_kl = torch.stack(kl_losses).mean()
    total_loss = surrogate_loss + KL_COEFF * loss_kl

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    global_step += 1
    if global_step % UPDATE_INTERVAL == 0:
        old_model.load_state_dict(model.module.state_dict())

    step_log = {
        "prompt": prompt,
        "candidate_stories": candidate_stories,
        "rule_rewards": rule_rewards,
        "model_rewards": model_rewards,
        "combined_rewards": combined_rewards,
        "normalized_advantages": normalized_advantages,
        "dropped_candidate_index": worst_idx,
        "loss_surrogate": surrogate_loss.item(),
        "kl_loss": loss_kl.item(),
        "total_loss": total_loss.item(),
    }
    training_log.append(step_log)

    if dist.get_rank() == 0:
        console.print(f"[bold green]Step {global_step}[/bold green] - Prompt: {prompt}")
        for story in candidate_stories:
            console.print("[blue]Candidate Story:[/blue] " + story)
        console.print(f"[magenta]Rule Rewards:[/magenta] {rule_rewards}")
        console.print(f"[magenta]Model Rewards:[/magenta] {model_rewards}")
        console.print(f"[magenta]Combined Rewards:[/magenta] {combined_rewards}")
        console.print(f"[magenta]Dropped Candidate Index:[/magenta] {worst_idx}")
        console.print(f"[yellow]Loss Surrogate:[/yellow] {surrogate_loss.item():.4f}, [yellow]KL Loss:[/yellow] {loss_kl.item():.4f}, [yellow]Total Loss:[/yellow] {total_loss.item():.4f}\n")
    
    return global_step

# ---------------------
# Main Training Function
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="Pretrained model name or path")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger.info(f"Rank {dist.get_rank()} using device: {device}")

    # Load main story generation model and tokenizer
    model_name = args.model_name
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model.train()
    model = torch.nn.parallel.DistributedDataParallel(base_model, device_ids=[local_rank])

    # Create a reference model for gradient-free evaluations (old_model is not DDP-wrapped)
    old_model = copy.deepcopy(model.module).to(device)
    old_model.eval()

    optimizer = Adam(model.parameters(), lr=LR)

    training_log = []
    global_step = 0
    console.print(f"[bold green]Starting training for {TOTAL_TRAIN_STEPS} steps.[/bold green]")

    pbar = tqdm(total=TOTAL_TRAIN_STEPS, disable=(dist.get_rank() != 0))
    while global_step < TOTAL_TRAIN_STEPS:
        # Generate a creative prompt using the same model (old_model) in gradient-free mode.
        creative_prompt = generate_creative_prompt(tokenizer, old_model, device)
        # Augment the prompt for story generation
        augmented_prompt = augment_prompt(creative_prompt)
        global_step = training_step(augmented_prompt, tokenizer, model, old_model, optimizer, device, training_log, global_step)
        pbar.update(1)

    pbar.close()

    if dist.get_rank() == 0:
        with open("training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)
        console.print("[bold green]Training log saved to training_log.json[/bold green]")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

