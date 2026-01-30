"""
Optimized Rollout Function implementation for the Alfworld Environment Task.
This implementation includes:
- Full episode training (all turns)
- Intermediate reward shaping for better credit assignment
- Robust action parsing with validation
- Per-prompt game sampling for diversity
- Enhanced reward calculation with progress tracking
- Better episode tracking and state management

Read more about rollout functions here: https://huggingface.co/docs/trl/main/en/openenv
"""

# NOTE: Keeping the original TRL helper for rollouts; server mode is configured via GRPOConfig.
# def _generate_rollout_completions(prompts: list, trainer, *, as_chat: bool | None = None) -> list[dict[str, list]]:
#     from trl.experimental.openenv import generate_rollout_completions as colocate_generate
#     return colocate_generate(trainer, prompts=prompts, as_chat=as_chat)


def extract_and_validate_action(completion_text: str, available_actions: list[str]) -> str:
    """
    Extract and validate action from completion text with multiple strategies.
    Returns validated action or empty string if no valid action found.
    """
    if not available_actions:
        return ""
    
    # Clean completion text
    action_candidate = completion_text.strip()
    if action_candidate.endswith("</s>"):
        action_candidate = action_candidate[:-5].strip()
    
    # Strategy 1: Extract after "Action:" marker
    if "Action:" in action_candidate:
        action_candidate = action_candidate.split("Action:")[-1].strip()
        # Remove any remaining "Thought:" content
        if "Thought:" in action_candidate:
            action_candidate = action_candidate.split("Thought:")[-1].strip()
    
    # Strategy 2: Direct match
    if action_candidate in available_actions:
        return action_candidate
    
    # Strategy 3: Case-insensitive match
    action_lower = action_candidate.lower()
    for avail_action in available_actions:
        if avail_action.lower() == action_lower:
            return avail_action
    
    # Strategy 4: Substring match (action contains available action)
    for avail_action in available_actions:
        if avail_action.lower() in action_lower or action_lower in avail_action.lower():
            return avail_action
    
    # Strategy 5: Fuzzy matching (simple word overlap)
    action_words = set(action_lower.split())
    best_match = None
    best_score = 0
    for avail_action in available_actions:
        avail_words = set(avail_action.lower().split())
        overlap = len(action_words & avail_words)
        if overlap > best_score and overlap > 0:
            best_score = overlap
            best_match = avail_action
    
    if best_match:
        return best_match
    
    # Fallback: return first available action (will likely trigger "Nothing happens")
    return available_actions[0] if available_actions else ""


def calculate_shaped_reward(
    observation: str,
    previous_observation: str,
    step_reward: float,
    done: bool,
    solved: bool,
    invalid_count: int,
    turn_number: int,
    max_turns: int
) -> float:
    """
    Calculate shaped reward with intermediate signals for better credit assignment.
    """
    reward = 0.0
    
    # Base reward from environment
    reward += step_reward
    
    # Reward for valid actions (not "Nothing happens")
    if "Nothing happens" not in observation and "Nothing happened" not in observation:
        reward += 0.01
    
    # Reward for state changes (new observations indicate progress)
    if observation != previous_observation and previous_observation:
        reward += 0.02
    
    # Reward for task-relevant keywords (indicates meaningful actions)
    task_keywords = ["pick up", "put", "open", "close", "go to", "examine", "take", "drop", "use"]
    observation_lower = observation.lower()
    if any(keyword in observation_lower for keyword in task_keywords):
        reward += 0.03
    
    # Reward for progress indicators
    progress_indicators = ["successfully", "found", "obtained", "placed", "moved"]
    if any(indicator in observation_lower for indicator in progress_indicators):
        reward += 0.05
    
    # Time efficiency bonus (reward faster completion)
    if solved:
        efficiency_bonus = (1.0 - (turn_number / max_turns)) * 0.3
        reward += efficiency_bonus + 1.0  # Large completion bonus
    else:
        # Small penalty for invalid actions
        reward -= 0.01 * invalid_count
    
    return max(reward, -0.5)  # Cap negative rewards


def alfworld_rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json

    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early

    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(alfworld_rollout_first_prompt_and_completion, "initialized", False):
        # Get local rank
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Get env server for that local rank
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
        
        # Determine endpoint
        if not server_list:
            # Fallback (though likely fatal for the task)
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        # Store endpoint on the function to avoid re-parsing
        alfworld_rollout_first_prompt_and_completion.base_url = base_url
        
        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing AlfWorld environment on rank {rank} at {base_url}...")
            create_res = requests.post(f"{base_url}/create", timeout=300)
            create_res.raise_for_status()
            # Store env_id on the function
            alfworld_rollout_first_prompt_and_completion.env_id = create_res.json()["id"]
            alfworld_rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. ID: {alfworld_rollout_first_prompt_and_completion.env_id}")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    # Retrieve static variables
    env_id = alfworld_rollout_first_prompt_and_completion.env_id
    env_endpoint = alfworld_rollout_first_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []
    all_episode_action_masks: list[list[int]] = []

    tokenizer = trainer.processing_class
    DATA_LEN = 2500
    TIMEOUT = 2400

    # Hardcoded System Prompt (ReAct)
    conversation_start = [
        {
            "from": "human",
            "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
        },
        {
            "from": "gpt",
            "value": "OK. I'll follow your instructions and try my best to solve the task.",
        }
    ]

    # --- 3. Batch Loop ---
    # OPTIMIZATION: Sample different game_id per prompt for better diversity
    for i, prompt in enumerate(prompts):
        # Sample different game for each prompt to increase episode diversity
        game_id = random.randint(0, DATA_LEN - 1)
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []
        prev_full_ids: list[int] | None = None
        invalid_count = 0
        valid_action_count = 0
        done = False
        solved = False
        turn_number = 0
        previous_observation = ""
        step_rewards = []  # Track rewards per step for shaping
        
        # --- Reset Environment (POST /reset) ---
        # Reuse existing env_id, just change the game
        payload = {"id": env_id, "game": game_id, "world_type": "Text"}
        
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            
            # Construct Initial Observation
            current_observation = reset_data["observation"]
            current_available_actions = reset_data["available_actions"]
            formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
            previous_observation = current_observation  # Initialize for reward shaping
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []
        for message in conversation_start:
            if message["from"] == "human":
                messages.append({"role": "user", "content": message["value"]})
            elif message["from"] == "gpt":
                messages.append({"role": "assistant", "content": message["value"]})
        
        messages.append({"role": "user", "content": formatted_observation})
        
        # Initialize previous_observation for reward shaping (if not already set)
        if not previous_observation:
            previous_observation = current_observation

        # --- Interaction Loop ---
        while not done and (turn_number < max_turns):
            # Generate Rollout Completion
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check if prompt exceeds max length - end episode early to prevent context overflow
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}, ending episode early")
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # BPE mismatch - tokenizer produced different IDs for same prefix text
                    # Graceful fallback: skip delta masking for this turn, just add completion
                    print(
                        f"Warning: BPE mismatch at turn {turn_number} (expected prefix {len(prev_full_ids)}, "
                        f"got {len(prompt_ids)} tokens). Skipping delta mask for this turn."
                    )
                    # Reset prev_full_ids to current prompt to try to recover alignment
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
                    if delta_prompt_ids:
                        episode_completion_ids.extend(delta_prompt_ids)
                        episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                        episode_action_mask.extend([0] * len(delta_prompt_ids))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse and Validate Action (OPTIMIZED) ---
            # Get available actions from previous step (or initial reset)
            action_to_send = extract_and_validate_action(completion_text, current_available_actions)
            
            # Track action validity
            is_valid_action = action_to_send in current_available_actions if action_to_send else False
            
            # --- Step Environment (POST /step) ---
            step_reward = 0.0
            step_done = False
            step_state = ""

            try:
                step_payload = {"id": env_id, "action": action_to_send}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()

                # Extract response data
                step_state = step_data["observation"]
                step_reward = step_data["reward"]
                step_done = step_data["done"]
                current_available_actions = step_data["available_actions"]
                
                # Format next observation
                formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
                
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + (formatted_observation if 'formatted_observation' in locals() else "")
                step_reward = 0.0
                step_done = False
                step_state = "Error occurred"

            # Update Loop State
            if step_done and step_reward > 0:
                solved = True

            # Track action validity
            if "Nothing happens" in step_state or "Nothing happened" in step_state:
                invalid_count += 1
            else:
                valid_action_count += 1
            
            # Calculate shaped reward for this step
            shaped_step_reward = calculate_shaped_reward(
                step_state,
                previous_observation,
                step_reward,
                step_done,
                solved,
                invalid_count,
                turn_number,
                max_turns
            )
            step_rewards.append(shaped_step_reward)
            previous_observation = step_state
            
            done = step_done

            if not done:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

        # OPTIMIZED: Use shaped reward with intermediate signals
        # Sum all step rewards (already includes shaping)
        if step_rewards:
            train_reward = sum(step_rewards)
        else:
            # Fallback to simple reward if no steps
            train_reward = (1.0 if solved else 0.0) - 0.01 * float(invalid_count)
        
        # Additional bonus for action efficiency
        total_actions = valid_action_count + invalid_count
        if total_actions > 0:
            action_efficiency = valid_action_count / total_actions
            train_reward += action_efficiency * 0.1  # Up to 0.1 bonus for efficiency
        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)
        all_episode_action_masks.append(episode_action_mask)

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards,
        "action_mask": all_episode_action_masks
    }

def alfworld_rollout_reward_func(completions, **kwargs):
    """
    Enhanced reward function that extracts shaped rewards from rollout function.
    The rewards are already shaped with intermediate signals in the rollout function.
    """
    rewards = kwargs.get("env_rewards") if kwargs else None
    if rewards is not None:
        # Ensure all rewards are floats and handle any edge cases
        return [float(r) for r in rewards]
    else:
        # Return zero rewards if not provided (shouldn't happen in normal operation)
        return [0.0] * len(completions)