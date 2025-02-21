import os
import copy
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------
# Setup Logging
# ---------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------
# Hyperparameters & Defaults
# ---------------------
GROUP_SIZE = 3               # Generate 3 candidate summaries per prompt
MAX_GEN_LEN = 150            # Maximum tokens to generate
EPSILON = 0.2                # Clipping parameter for GRPO
KL_COEFF = 0.01              # Coefficient for KL penalty
LR = 1e-5                    # Learning rate
UPDATE_INTERVAL = 2          # Update reference model every 2 training steps
NUM_RANKING_PASSES = 3       # Use three ranking passes to reduce variance
NUM_EPOCHS = 3               # Number of training epochs

# ---------------------
# Self-Critic Ranking Function with Multiple Passes & Shuffling
# ---------------------
def self_critic_rank(prompt: str, completions: list, tokenizer, model, device, num_passes: int = NUM_RANKING_PASSES) -> list:
    """
    Ranks candidate summaries using multiple passes with shuffled orders.
    The model is instructed to rank summaries based on clarity, conciseness,
    and coverage of key points.
    """
    num_candidates = len(completions)
    aggregated_scores = np.zeros(num_candidates, dtype=np.float32)

    for _ in range(num_passes):
        # Create a random permutation of candidate indices
        perm = np.random.permutation(num_candidates)
        shuffled_completions = [completions[i] for i in perm]

        ranking_prompt = (
            "You are a summarization quality evaluator. Your task is to rank the following summaries "
            "based on their clarity, conciseness, and how well they capture the key points of the document.\n\n"
            "Summaries to rank:\n"
        )
        for idx, comp in enumerate(shuffled_completions):
            ranking_prompt += f"{idx+1}. {comp}\n"
        ranking_prompt += (
            f"\nProvide ONLY a comma-separated list of {num_candidates} numbers ranking the summaries "
            "from best (first) to worst."
        )

        with torch.no_grad():
            inputs = tokenizer(ranking_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            ranking_output = model.module.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        ranking_text = tokenizer.decode(ranking_output[0].cpu(), skip_special_tokens=True).splitlines()[0].strip()

        # Parse ranking output; if parsing fails, fallback to default ranking
        try:
            ranking_numbers = [int(x.strip()) for x in ranking_text.split(",") if x.strip().isdigit()]
            if len(ranking_numbers) != num_candidates:
                ranking_numbers = list(range(1, num_candidates + 1))
        except Exception as e:
            logger.warning(f"Ranking parse error: {e}. Using default ranking.")
            ranking_numbers = list(range(1, num_candidates + 1))

        # Map the ranking back to the original candidate order.
        # The ranking_numbers list corresponds to positions in the shuffled list.
        scores = np.zeros(num_candidates, dtype=np.float32)
        # For each position in the ranking output, assign a score (higher is better)
        for rank_pos, candidate_label in enumerate(ranking_numbers):
            # candidate_label is the number in the shuffled list (1-indexed)
            original_idx = perm[candidate_label - 1]
            scores[original_idx] = num_candidates - rank_pos  # best gets highest score

        aggregated_scores += scores

    # Average the scores over all passes to get final reward estimates
    final_scores = aggregated_scores / num_passes


    return final_scores.tolist()

# ---------------------
# Log Probability Function
# ---------------------
def compute_log_prob(model, input_ids, gen_ids):
    """
    Computes the total log probability of the generated tokens.
    """
    full_ids = torch.cat([input_ids, gen_ids], dim=-1)
    outputs = model(full_ids)
    logits = outputs.logits
    L_prompt = input_ids.shape[-1]
    # Select logits corresponding to generated tokens (shifted by one)
    gen_logits = logits[:, L_prompt-1:-1, :]
    log_probs = F.log_softmax(gen_logits, dim=-1)
    gen_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
    total_log_prob = gen_log_probs.sum(dim=-1)
    return total_log_prob

# ---------------------
# Dataset Preparation: Using XSum for Summarization
# ---------------------
class PromptDataset(Dataset):
    def __init__(self, examples):
        # Use the "document" field as input text
        self.documents = [ex["document"] for ex in examples]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]

def augment_prompt(document: str) -> str:
    """
    Augments the raw document with a summarization instruction.
    """
    prompt = (
        "You are an AI summarization expert. Your task is to generate a concise, high-quality summary "
        "of the following document. The summary should be a single, coherent paragraph capturing the key points.\n\n"
        "Document:\n" + document + "\n\nSummary:"
    )
    return prompt

# ---------------------
# Training Step Function
# ---------------------
def training_step(prompt: str, tokenizer, model, old_model, optimizer, device, training_log, global_step):
    # Tokenize the prompt with truncation to save memory
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)

    candidate_gen_ids = []
    completions = []
    # Generate GROUP_SIZE candidate summaries
    for _ in range(GROUP_SIZE):
        gen_output = old_model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + MAX_GEN_LEN,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        # Extract generated tokens (after the prompt)
        gen_ids = gen_output[:, input_ids.shape[-1]:].to(device)
        candidate_gen_ids.append(gen_ids)

        # Decode and extract only the generated summary
        full_text = tokenizer.decode(gen_ids[0].cpu(), skip_special_tokens=True)
        # In case the generation includes the prompt artifact, split on "Summary:" marker
        summary = full_text.split("Summary:")[-1].strip()
        completions.append(summary)

    # Free GPU memory if needed
    torch.cuda.empty_cache()

    # Use the ranking function to obtain reward scores for each candidate
    rewards = self_critic_rank(prompt, completions, tokenizer, model, device)
    rewards_np = np.array(rewards)
    mean_reward = rewards_np.mean()
    std_reward = rewards_np.std() + 1e-8
    normalized_advantages = [(r - mean_reward) / std_reward for r in rewards]

    # Optionally drop the worst candidate (here, the one with minimum raw reward)
    worst_idx = int(np.argmin(rewards))
    filtered_advantages = [normalized_advantages[i] for i in range(GROUP_SIZE) if i != worst_idx]
    filtered_gen_ids = [candidate_gen_ids[i] for i in range(GROUP_SIZE) if i != worst_idx]

    candidate_objs = []
    kl_losses = []
    # Compute the surrogate objective and KL penalty for each remaining candidate
    for gen_ids, adv in zip(filtered_gen_ids, filtered_advantages):
        logp_new = compute_log_prob(model, input_ids, gen_ids)
        with torch.no_grad():
            logp_old = compute_log_prob(old_model, input_ids, gen_ids)
        ratio = torch.exp(logp_new - logp_old)
        clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
        adv_tensor = torch.tensor(adv, device=device)
        if adv_tensor >= 0:
            candidate_obj = torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor)
        else:
            candidate_obj = torch.max(ratio * adv_tensor, clipped_ratio * adv_tensor)
        candidate_objs.append(candidate_obj)

        # Compute KL divergence between new and old model outputs
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
    # Periodically update the reference (old) model with the current policy
    if global_step % UPDATE_INTERVAL == 0:
        old_model.load_state_dict(model.module.state_dict())

    # Log training metrics
    step_log = {
        "prompt": prompt,
        "raw_rewards": rewards,
        "dropped_candidate_index": worst_idx,
        "normalized_advantages": normalized_advantages,
        "filtered_advantages": filtered_advantages,
        "loss_surrogate": surrogate_loss.item(),
        "kl_loss": loss_kl.item(),
        "total_loss": total_loss.item(),
        "completions": completions if global_step % 3 == 0 else None,
    }
    training_log.append(step_log)

    if dist.get_rank() == 0:
        logger.info(f"Step {global_step} - Prompt: {prompt}")
        for comp in completions:
            logger.info("Candidate Summary: " + comp)
        logger.info(f"Raw Rewards: {rewards}")
        logger.info(f"Dropped Candidate Index: {worst_idx}")
        logger.info(f"Normalized Advantages: {normalized_advantages}")
        logger.info(f"Loss Surrogate: {surrogate_loss.item():.4f}, KL Loss: {loss_kl.item():.4f}, Total Loss: {total_loss.item():.4f}\n")

    return global_step

# ---------------------
# Main Training Function
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="Name or path of the pretrained model")
    args = parser.parse_args()

    # ---------------------
    # Distributed Setup
    # ---------------------
    dist.init_process_group(backend="nccl")
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger.info(f"Rank {dist.get_rank()} using device: {device}")

    # ---------------------
    # Load Dataset and Prepare Prompts (XSum Summarization)
    # ---------------------
    dataset = load_dataset("xsum", split="train")
    sampled_dataset = dataset.shuffle(seed=42).select(range(1000))
    prompt_dataset = PromptDataset(sampled_dataset)
    # Augment each document with a summarization instruction
    prompts = [augment_prompt(doc) for doc in prompt_dataset.documents]

    sampler = DistributedSampler(prompts)
    dataloader = DataLoader(prompts, sampler=sampler, batch_size=1, shuffle=False)

    # ---------------------
    # Load Model and Tokenizer; Wrap with DDP
    # ---------------------
    model_name = args.model_name
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model.train()
    model = torch.nn.parallel.DistributedDataParallel(base_model, device_ids=[local_rank])

    # Create a reference model for computing old probabilities
    old_model = copy.deepcopy(model.module).to(device)
    old_model.eval()

    optimizer = Adam(model.parameters(), lr=LR)

    # ---------------------
    # Training Loop
    # ---------------------
    training_log = []
    global_step = 0
    total_steps = NUM_EPOCHS * len(dataloader)
    if dist.get_rank() == 0:
        logger.info(f"Training for {NUM_EPOCHS} epochs (~{total_steps} steps).")

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        if dist.get_rank() == 0:
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for prompt_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(dist.get_rank() != 0)):
            # Since batch_size=1, extract the single prompt string
            prompt_text = prompt_batch[0]
            global_step = training_step(prompt_text, tokenizer, model, old_model, optimizer, device, training_log, global_step)

    # ---------------------
    # Save Training Log (Only Rank 0)
    # ---------------------
    if dist.get_rank() == 0:
        with open("training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)
        logger.info("Training log saved to training_log.json")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

