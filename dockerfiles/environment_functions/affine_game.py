"""
Optimized Rollout Function implementation for the Affine GAME Environment Task.
This implementation includes:
- Full episode training (all turns)
- Intermediate reward shaping for better credit assignment
- Robust numeric action parsing with validation
- Per-prompt game sampling for diversity
- Enhanced reward calculation with progress tracking
- Action mask support for cleaner training
- Better episode tracking and state management

Notes:
With the Affine GAME environment when you reset to start a new game you have to choose an 'opponent' type to train against.
Your two options are 'random' and 'mcts'.
Miners are free to choose which opponent type they train against.
"""

def extract_and_validate_numeric_action(completion_text: str, valid_action_range: tuple[int, int] = None) -> int | None:
    """
    Extract numeric action ID from completion text with validation.
    Returns action ID or None if invalid.
    
    This is critical for chat-capable models that may output verbose responses.
    """
    # Clean completion text
    action_candidate = completion_text.strip()
    if action_candidate.endswith("</s>"):
        action_candidate = action_candidate[:-5].strip()
    
    # Strategy 1: Extract after "Action:" marker
    if "Action:" in action_candidate:
        action_candidate = action_candidate.split("Action:")[-1].strip()
        # Remove any "Thought:" content
        if "Thought:" in action_candidate:
            action_candidate = action_candidate.split("Thought:")[-1].strip()
    
    # Strategy 2: Extract first number found (readable approach without regex)
    # Look for the first sequence of digits, optionally starting with a minus sign
    number_chars = []
    found_minus = False
    found_digit = False
    
    for char in action_candidate:
        if char == '-' and not found_digit and not found_minus:
            # Found a minus sign at the start of a potential number
            number_chars.append(char)
            found_minus = True
        elif char.isdigit():
            # Found a digit - add it and mark that we're building a number
            number_chars.append(char)
            found_digit = True
        elif found_digit:
            # We were building a number but hit a non-digit - stop here
            break
        elif found_minus and not char.isdigit():
            # We had a minus but no digit after it - reset
            number_chars = []
            found_minus = False
    
    # If we found a number, try to convert it
    if number_chars and found_digit:
        number_str = ''.join(number_chars)
        try:
            action_id = int(number_str)
            
            # Validate range if provided
            if valid_action_range:
                min_action, max_action = valid_action_range
                if min_action <= action_id <= max_action:
                    return action_id
                else:
                    print(f"Warning: Action {action_id} out of range [{min_action}, {max_action}]")
                    return None
            
            return action_id
        except ValueError:
            pass
    
    # Strategy 3: Try direct conversion of the entire cleaned text
    try:
        action_id = int(action_candidate)
        if valid_action_range:
            min_action, max_action = valid_action_range
            if min_action <= action_id <= max_action:
                return action_id
        return action_id
    except ValueError:
        pass
    
    return None


def calculate_game_shaped_reward(
    step_reward: float,
    previous_reward: float,
    cumulative_reward: float,
    done: bool,
    turn_number: int,
    max_turns: int,
    game_type: str = "goofspiel"
) -> float:
    """
    Calculate shaped reward for game environments with intermediate signals.
    
    This provides better credit assignment for chat-capable models by:
    - Rewarding intermediate progress
    - Encouraging efficient play
    - Providing signals throughout the episode, not just at the end
    """
    reward = 0.0
    
    # Base reward from environment (already includes game score)
    reward += step_reward
    
    # Reward for improving position (positive reward change)
    if step_reward > previous_reward:
        reward += 0.05  # Bonus for improving
    
    # Reward for maintaining advantage (positive cumulative)
    if cumulative_reward > 0:
        reward += 0.02  # Small bonus for being ahead
    
    # Time efficiency bonus (faster wins are better)
    if done and step_reward > 0:  # Won the game
        efficiency_bonus = (1.0 - (turn_number / max_turns)) * 0.2
        reward += efficiency_bonus
    
    # Penalty for taking too long (encourage decisive play)
    if turn_number > max_turns * 0.8:  # Last 20% of turns
        reward -= 0.01
    
    # Goofspiel-specific shaping: reward strategic play
    if game_type == "goofspiel":
        # Reward for maintaining positive cumulative score
        if cumulative_reward > 0.5:  # Strong lead
            reward += 0.03
        elif cumulative_reward > 0:  # Small lead
            reward += 0.01
    
    return reward


def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json

    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early

    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "goofspiel"
    
    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(rollout_first_prompt_and_completion, "initialized", False):
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
        rollout_first_prompt_and_completion.base_url = base_url
        
        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing environment on rank {rank} at {base_url}...")
            payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
            create_res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
            create_res.raise_for_status()
            rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. Rank: {rank}.")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    # Retrieve static variables
    env_endpoint = rollout_first_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []
    all_episode_action_masks: list[list[int]] = []

    tokenizer = trainer.processing_class
    TIMEOUT = 2400

    # --- 3. Batch Loop ---
    # OPTIMIZED: Sample different game_id per prompt for better diversity
    for i, prompt in enumerate(prompts):
        # Per-prompt game sampling for diversity
        game_id = random.randint(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1])
        
        # OPTIMIZED: Vary opponent type for diversity (optional)
        # opponent_type = random.choice(["random", "mcts"]) if i % 2 == 0 else "mcts"
        opponent_type = "mcts"  # Keep mcts for now, can be varied
        
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []
        prev_full_ids: list[int] | None = None
        done = False
        solved = False
        turn_number = 0
        valid_action_count = 0
        invalid_action_count = 0
        step_rewards = []  # Track rewards per step for shaping
        cumulative_reward = 0.0
        previous_step_reward = 0.0
        
        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": random.randint(0, 1000000), "opponent": opponent_type}
        
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]
            
            # Get episode id for rest of interactions
            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            current_observation = result_block.get("observation", "")
            format_instructions = 'Your output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
            current_observation += format_instructions

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []
        messages.append({"role": "user", "content": current_observation})

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

            # OPTIMIZED: Full episode training (not just first turn)
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
                    # Add delta (observations between turns) as non-action tokens
                    delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
                    if delta_prompt_ids:
                        episode_completion_ids.extend(delta_prompt_ids)
                        episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                        episode_action_mask.extend([0] * len(delta_prompt_ids))  # Not actions
                    prev_full_ids = prompt_ids.copy()

            # Add completion (action tokens)
            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))  # These are actions
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse and Validate Action (OPTIMIZED) ---
            # Extract numeric action ID with validation
            # This is critical for chat-capable models that may output verbose responses
            action_id = extract_and_validate_numeric_action(completion_text)
            
            if action_id is not None:
                valid_action_count += 1
                action_to_send = str(action_id)
            else:
                invalid_action_count += 1
                # Fallback to "0" or first available action
                action_to_send = "0"
                print(f"Warning: Could not extract valid action from: {completion_text[:100]}")
            
            # --- Step Environment (POST /step) ---
            step_reward = 0.0
            step_done = False
            step_state = ""

            try:
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data
                step_state = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0)
                step_done = step_block.get("done", False)
                
                # Format next observation
                formatted_observation = step_state
                
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + (formatted_observation if formatted_observation else "")
                step_reward = -0.1  # Penalty for invalid action
                step_done = False
                step_state = "Error occurred"

            # Update loop state
            # In games like Goofspiel, done=True means game ended (win or loss)
            # Reward can be positive (win) or negative (loss)
            if step_done:
                solved = True  # Game completed (win or loss)

            # OPTIMIZED: Calculate shaped reward for this step
            cumulative_reward += step_reward
            shaped_step_reward = calculate_game_shaped_reward(
                step_reward,
                previous_step_reward,
                cumulative_reward,
                step_done,
                turn_number,
                max_turns,
                selected_game
            )
            step_rewards.append(shaped_step_reward)
            previous_step_reward = step_reward
            
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
            train_reward = step_reward if done else 0.0
        
        # Additional bonus for action efficiency
        total_actions = valid_action_count + invalid_action_count
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
        "action_mask": all_episode_action_masks  # OPTIMIZED: Add action mask
    }

def rollout_reward_func(completions, **kwargs):
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
