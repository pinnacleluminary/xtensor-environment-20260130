import os
import re
import random
import json
import requests
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from trl.experimental.openenv import generate_rollout_completions


# ============================================================
# Module-Level Constants (shared across all rollout functions)
# ============================================================

GAMES_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}

# Games where strategy forcing is supported
GAMES_WITH_STRATEGY_FORCING = {"gin_rummy"}

# Per-game system prompts
# NOTE: Server already sends full game rules via agent.get_rules() in /reset observation.
# System prompts only need output format instructions — do NOT duplicate rules here.
GAME_SYSTEM_PROMPTS = {
    "gin_rummy": (
        "You are playing Gin Rummy.\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "52 -> Draw upcard": respond "52"\n'
        '- For action "53 -> Draw stock": respond "53"\n'
        '- For action "5 -> discard": respond "5"'
    ),
    "goofspiel": (
        "You are playing Goofspiel (Game of Pure Strategy).\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "0 -> bid card 1": respond "0"\n'
        '- For action "7 -> bid card 8": respond "7"'
    ),
    "liars_dice": (
        "You are playing Liar's Dice.\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "0 -> 1-1" (bid 1 die showing 1): respond "0"\n'
        '- For action "29 -> Liar": respond "29"'
    ),
    "leduc_poker": (
        "You are playing Leduc Poker.\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "0 -> Fold": respond "0"\n'
        '- For action "1 -> Call": respond "1"\n'
        '- For action "2 -> Raise": respond "2"'
    ),
    "othello": (
        "You are playing Othello (Reversi).\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "19 -> d4": respond "19"\n'
        '- For action "37 -> f5": respond "37"'
    ),
    "backgammon": (
        "You are playing Backgammon.\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "0 -> 8/5/3": respond "0"\n'
        '- For action "12 -> 6/3": respond "12"'
    ),
    "hex": (
        "You are playing Hex.\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "0 -> a1": respond "0"\n'
        '- For action "12 -> c3": respond "12"'
    ),
    "clobber": (
        "You are playing Clobber.\n\n"
        "# Output Format\n"
        "You must respond with ONLY the action ID (a single number).\n"
        "Do NOT include descriptions or explanations.\n\n"
        "Examples:\n"
        '- For action "0 -> 0 0": respond "0"\n'
        '- For action "5 -> 2 3": respond "5"'
    ),
}

# Fallback prompt for games without a specific prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are playing a strategic game.\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Choose from the legal actions provided. Respond with just the ID number."
)

# Per-game strategy hints (aligned with agent get_rules() and game mechanics)
GAME_STRATEGY_HINTS = {
    "gin_rummy": (
        "\n\n# Strategy Tips\n"
        "- Draw phase: Use 52 for upcard, 53 for stock pile\n"
        "- Build melds: sets (3+ same rank) and runs (3+ consecutive same suit)\n"
        "- Track opponent discards to avoid feeding their melds\n"
        "- Knock when deadwood \u2264 knock_card value (check game variant)\n"
        "- Gin (0 deadwood) = 25-point bonus"
    ),
    "goofspiel": (
        "\n\n# Strategy Tips\n"
        "- Save high bid cards for high prize cards\n"
        "- If bids tie, prize is discarded (no one gets points)\n"
        "- Consider what opponent might bid based on prizes won so far\n"
        "- Track remaining bid cards for both players"
    ),
    "liars_dice": (
        "\n\n# Strategy Tips\n"
        "- 6s are WILD: count as ANY face value in all bids\n"
        "- Bid based on your own dice + expected opponent dice\n"
        "- Higher bid = same face higher quantity, OR same quantity higher face\n"
        "- Call Liar when bid exceeds realistic total dice count"
    ),
    "leduc_poker": (
        "\n\n# Strategy Tips\n"
        "- Hand ranking: Pair (private+public match) > High card (K>Q>J)\n"
        "- Round 1 raise = 2 chips, Round 2 raise = 4 chips\n"
        "- Max 2 raises per round\n"
        "- Raise with K, consider folding J vs opponent raise"
    ),
    "othello": (
        "\n\n# Strategy Tips\n"
        "- Corners cannot be flipped — prioritize capturing them\n"
        "- Avoid placing on edges adjacent to empty corners\n"
        "- Flank opponent discs horizontally, vertically, or diagonally\n"
        "- Must flip at least 1 disc; if no valid move, pass"
    ),
    "backgammon": (
        "\n\n# Strategy Tips\n"
        "- x = your checkers, o = opponent's checkers\n"
        "- Avoid leaving blots (single checkers) exposed to hits\n"
        "- Make points (2+ checkers) to block opponent movement\n"
        "- Bar checkers must re-enter before other moves\n"
        "- Bear off efficiently once all checkers are in home board"
    ),
    "hex": (
        "\n\n# Strategy Tips\n"
        "- Red (x) connects top-left to bottom-right\n"
        "- Blue (o) connects top-right to bottom-left\n"
        "- Control center for flexible connections\n"
        "- No draws possible — someone must win"
    ),
    "clobber": (
        "\n\n# Strategy Tips\n"
        "- Every move MUST capture an adjacent opponent piece\n"
        "- Keep your pieces connected to maintain move options\n"
        "- Isolate opponent pieces so they run out of captures\n"
        "- Player with no legal moves loses"
    ),
}

DEFAULT_STRATEGY_HINTS = (
    "\n\n# Strategy Tips\n"
    "- Think carefully about the game state before choosing\n"
    "- Consider your opponent's likely moves\n"
    "- Try to maximize your score while minimizing risk"
)


def extract_and_format_observation(obs_text):
    """
    Extract and format observation to match evaluation format.
    Reconstructs legal actions from player hand.
    
    Args:
        obs_text: Raw observation text from result_block
        
    Returns:
        Formatted observation string matching evaluation format
    """
    
    # Case 1: Error message with legal actions already present
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        # Already formatted correctly, return as-is
        return obs_text
    
    # Case 2: Normal observation - extract state and reconstruct legal actions
    
    # Extract everything after "Current State:"
    state_match = re.search(
        r'Current State:\n(.*)',
        obs_text,
        re.DOTALL
    )
    
    if not state_match:
        # Fallback: return original if no "Current State:" found
        return obs_text
    
    state_text = state_match.group(0)
    
    # Remove "Waiting for Player -2 to move..." if present
    state_text = re.sub(
        r'\n\nWaiting for Player -2 to move\.\.\.$',
        '',
        state_text
    )
    
    # Detect player ID (look for "You are Player X")
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id = int(player_match.group(1)) if player_match else 0
    
    # Extract player hand to reconstruct legal actions
    hand_pattern = rf'P{player_id} hand: ([\d\s]+)'
    hand_match = re.search(hand_pattern, state_text)
    
    if not hand_match:
        # Can't find hand, return state without legal actions
        return state_text
    
    # Parse hand cards
    hand_str = hand_match.group(1).strip()
    cards = [int(card) for card in hand_str.split()]
    
    # Reconstruct legal actions
    # The action ID corresponds to the bid value - 1 (0-indexed)
    # But we need to map to the actual available cards
    legal_actions = []
    for i, card in enumerate(cards):
        # Action ID is the card value minus 1
        action_id = card - 1
        legal_actions.append(f"{action_id} -> [P{player_id}]Bid: {card}")
    
    # Format the complete observation
    formatted = state_text + "\n\nYou are Player " + str(player_id) + ".\nLegal Actions:\n"
    formatted += "\n".join(legal_actions)
    formatted += "\n\nYour choice (ID only):"
    
    return formatted


def extract_prize_card(obs_text):
    """
    Extract the current prize card value from observation.
    
    Args:
        obs_text: Observation text
        
    Returns:
        Prize card value (int) or None if not found
    """
    # Look for "Point card: X"
    match = re.search(r'Current point card:\s*(\d+)', obs_text)
    if match:
        return int(match.group(1))
    return None


def extract_bid_from_action(action_text, obs_text):
    """
    Extract the bid value from the action.
    
    Args:
        action_text: The action string (should be action ID)
        obs_text: The observation to help parse legal actions
        
    Returns:
        Bid card value (int) or None if cannot parse
    """
    try:
        action_id = int(action_text.strip())
        # The bid value is action_id + 1
        return action_id + 1
    except Exception:
        return None
    

def get_hand_cards(observation_text: str, player_id: int = 0) -> list[int]:
    """Count how many cards remain in the player's hand."""
    pattern = rf"P{player_id} hand:\s*([\d ]+)"
    match = re.search(pattern, observation_text)
    if not match:
        return []
    string_cards = match.group(1).strip().split()
    return [int(card) for card in string_cards]


REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

def remove_reasoning_tags(text: str) -> str:

    cleaned = text

    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]

        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]

    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


class FailureBuffer:
    """
    Replay buffer for failed/low-scoring tasks.
    Prioritizes difficult tasks for more practice.
    """
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.task_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
    
    def add(self, task_id: int, score: float, success: bool):
        """Add task result to buffer if failed or low score."""
        self.task_stats[task_id]["attempts"] += 1
        if success:
            self.task_stats[task_id]["successes"] += 1
        
        if not success or score < 0.5:
            self.buffer.append({
                "task_id": task_id,
                "score": score,
                "priority": 1.0 - score,
            })
    
    def sample(self, k: int = 1) -> list[int]:
        """Sample task IDs with priority weighting."""
        if not self.buffer:
            return []
        
        priorities = np.array([item["priority"] for item in self.buffer])
        total = priorities.sum()
        if total == 0:
            return []
        probs = priorities / total
        
        indices = np.random.choice(
            len(self.buffer),
            size=min(k, len(self.buffer)),
            replace=False,
            p=probs
        )
        
        return [self.buffer[i]["task_id"] for i in indices]
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        total_attempts = sum(s["attempts"] for s in self.task_stats.values())
        total_successes = sum(s["successes"] for s in self.task_stats.values())
        return {
            "buffer_size": len(self.buffer),
            "unique_tasks": len(self.task_stats),
            "total_attempts": total_attempts,
            "success_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
        }


class CurriculumScheduler:
    """
    Manages curriculum learning parameters throughout training.
    
    Features:
    - Turn-count curriculum (max_turn increases gradually)
    - Hint probability decay
    - Performance gating (won't advance if win rate too low)
    - Failure replay buffer (prioritizes difficult tasks)
    - Checkpoint persistence (save/load state)
    """
    def __init__(
        self,
        initial_max_turn=10,
        final_max_turn=50,
        rollouts_per_stage=1280,
        initial_hint_prob=0.50,
        final_hint_prob=0.0,
        warmup_rollouts=128,
        # Performance gating parameters
        progression_threshold=0.7,
        eval_window=100,
        # Failure replay parameters
        failure_buffer_size=500,
        failure_replay_prob=0.3,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        
        self.total_rollouts = 0
        
        # Performance gating
        self.progression_threshold = progression_threshold
        self.eval_window = eval_window
        self.recent_results = deque(maxlen=eval_window)
        self._max_achieved_stage = 0  # Highest stage reached by rollout count
        self._gated_stage = 0         # Actual stage (gated by performance)
        
        # Failure replay buffer
        self.failure_buffer = FailureBuffer(max_size=failure_buffer_size)
        self.failure_replay_prob = failure_replay_prob
        
    def get_max_turn(self):
        """Calculate current max_turn based on curriculum + performance gating."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        
        # Calculate stage by rollout count (time-based)
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        time_stage = adjusted_rollouts // self.rollouts_per_stage
        self._max_achieved_stage = time_stage
        
        # Performance gating: only advance if win rate >= threshold
        if len(self.recent_results) >= self.eval_window // 2:
            win_rate = sum(self.recent_results) / len(self.recent_results)
            if win_rate >= self.progression_threshold:
                # Allow advancement up to time_stage
                self._gated_stage = min(time_stage, self._gated_stage + 1)
            # else: keep current _gated_stage (don't advance)
        else:
            # Not enough data yet, follow time-based schedule
            self._gated_stage = time_stage
        
        current_max_turn = min(
            self.initial_max_turn + self._gated_stage,
            self.final_max_turn
        )
        return current_max_turn
    
    def get_hint_prob(self):
        """Calculate current hint probability based on curriculum."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        
        total_stages = self.final_max_turn - self.initial_max_turn
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)
        
        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)
    
    def update(self, task_id: int, score: float, success: bool):
        """
        Update curriculum with episode result.
        Feeds into performance gating and failure replay buffer.
        """
        self.recent_results.append(success)
        self.failure_buffer.add(task_id, score, success)
    
    def should_replay_failure(self) -> bool:
        """Check if this episode should be a failure replay."""
        return (random.random() < self.failure_replay_prob 
                and len(self.failure_buffer.buffer) > 0)
    
    def sample_failure_task(self) -> int | None:
        """Sample a task_id from the failure buffer."""
        tasks = self.failure_buffer.sample(k=1)
        return tasks[0] if tasks else None
    
    def step(self, num_rollouts=1):
        """Increment rollout counter."""
        self.total_rollouts += num_rollouts
        
    def get_status(self):
        """Get current curriculum status for logging."""
        win_rate = sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0.0
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(),
            "win_rate": win_rate,
            "gated_stage": self._gated_stage,
            "time_stage": self._max_achieved_stage,
            "failure_buffer_size": len(self.failure_buffer.buffer),
        }
    
    def get_state(self) -> dict:
        """Get state for checkpointing."""
        return {
            "total_rollouts": self.total_rollouts,
            "gated_stage": self._gated_stage,
            "max_achieved_stage": self._max_achieved_stage,
            "recent_results": list(self.recent_results),
            "failure_buffer_stats": self.failure_buffer.get_stats(),
        }
    
    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.total_rollouts = state.get("total_rollouts", 0)
        self._gated_stage = state.get("gated_stage", 0)
        self._max_achieved_stage = state.get("max_achieved_stage", 0)
        recent = state.get("recent_results", [])
        self.recent_results.clear()
        self.recent_results.extend(recent)


def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json

    games_to_task_id_range = GAMES_TO_TASK_ID_RANGE

    selected_game = getattr(trainer.args, "selected_game", "gin_rummy")
    
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

    tokenizer = trainer.processing_class
    TIMEOUT = 2400

    # --- 3. Batch Loop ---
    # We use a random game_id for the batch, or you could sample per item if preferred
    game_id = random.randint(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1])

    for i, prompt in enumerate(prompts):
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        done = False
        solved = False
        train_reward = 0
        turn_number = 0
        
        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}
        
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

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                episode_completion_ids = completion_ids
                episode_logprobs = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()
            
            # --- Step Environment (POST /step) ---

            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data
                step_state = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)
                
                # Format next observation
                formatted_observation = step_state
                
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation 
                step_reward = -0.01
                done = False

            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1
        
        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)

        

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards
    }


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for game environments.
    Uses full prompt and completion IDs with action masking.
    """
    # --- Constants ---
    STRATEGY_REWARD = 1.0
    INVALID_PENALTY = -0.1
    
    games_to_task_id_range = GAMES_TO_TASK_ID_RANGE

    selected_game = getattr(trainer.args, "selected_game", "gin_rummy")

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_last_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []  # list of dicts: {base_url}

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                # Initialize with a test reset to ensure server is ready
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
                res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
                res.raise_for_status()
                env_pool.append({"base_url": base_url})
                print(f"[INIT] Server {idx} ready")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        rollout_last_prompt_and_completion_parallelized_curriculum.rank = rank
        rollout_last_prompt_and_completion_parallelized_curriculum.env_pool = env_pool
        rollout_last_prompt_and_completion_parallelized_curriculum.num_servers = len(env_pool)
        rollout_last_prompt_and_completion_parallelized_curriculum.initialized = True
        rollout_last_prompt_and_completion_parallelized_curriculum.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        rollout_last_prompt_and_completion_parallelized_curriculum.generation_semaphore = Semaphore(1)
        rollout_last_prompt_and_completion_parallelized_curriculum.games_to_task_id_range = games_to_task_id_range
        rollout_last_prompt_and_completion_parallelized_curriculum.selected_game = selected_game
        
        # Initialize curriculum scheduler
        rollout_last_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=45,  # Gin Rummy: games can go up to 30-50 turns
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.45,  # Lower for complex game
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=50, rollouts_per_stage={trainer.args.rollouts_per_stage}, initial_hint_prob=0.50, final_hint_prob=0.0, warmup_rollouts={trainer.args.rollouts_per_stage}")

    # Retrieve static variables
    rank = rollout_last_prompt_and_completion_parallelized_curriculum.rank
    env_pool = rollout_last_prompt_and_completion_parallelized_curriculum.env_pool
    num_servers = rollout_last_prompt_and_completion_parallelized_curriculum.num_servers
    games_to_task_id_range = rollout_last_prompt_and_completion_parallelized_curriculum.games_to_task_id_range
    selected_game = rollout_last_prompt_and_completion_parallelized_curriculum.selected_game
    curriculum = rollout_last_prompt_and_completion_parallelized_curriculum.curriculum
    
    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    
    # Get current curriculum parameters
    total_rollouts = curriculum.total_rollouts
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    def run_single_prompt(index: int, prompt: str):
        # Generate a random game_id for this episode
        game_id = int(prompt)

        # Select server based on index and rank
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]
        done = False
        turn_number = 0
        target_training_turn = current_max_turn - 1
        
        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            # Get episode id for rest of interactions
            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            raw_observation = result_block.get("observation", "")
            formatted_observation = extract_and_format_observation(raw_observation)

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # First make system prompt
        system_prompt = GAME_SYSTEM_PROMPTS.get(selected_game, DEFAULT_SYSTEM_PROMPT)

        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = GAME_STRATEGY_HINTS.get(selected_game, DEFAULT_STRATEGY_HINTS)
            system_prompt += suggestion_prompt

        messages = [{"role": "system", "content": system_prompt}]

        # Strategy forcing (only for supported games)
        if selected_game in GAMES_WITH_STRATEGY_FORCING:
            while not done and (turn_number < target_training_turn):
                messages.append({"role": "user", "content": formatted_observation})

                hand_cards = get_hand_cards(formatted_observation)
                if len(hand_cards) <= 1:
                    target_training_turn = turn_number
                    break
                
                prize_card = extract_prize_card(formatted_observation)
                action_id = prize_card - 1

                messages.append({"role": "assistant", "content": str(action_id)})

                # --- Step Environment (POST /step) ---
                try:
                    formatted_observation = ""
                    step_payload = {"action": str(action_id), "episode_id": episode_id}
                    step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                    step_res.raise_for_status()
                    step_data = step_res.json()
                    step_block = step_data["result"]

                    # Extract response data
                    raw_observation = step_block.get("observation", "")
                    formatted_observation = extract_and_format_observation(raw_observation)
                    step_reward = step_block.get("reward", 0)
                    done = step_block.get("done", False)

                except Exception as e:
                    print(f"Step failed: {e}")
                    step_reward = -0.01
                    done = False

                turn_number += 1

            if done:
                print(
                    f"[GT] Game {game_id} ended during strategy forcing phase at turn {turn_number}. "
                    f"Returning fallback."
                )
                return index, None

        messages.append({"role": "user", "content": formatted_observation})

        with rollout_last_prompt_and_completion_parallelized_curriculum.generation_semaphore:
            rollout_out = generate_rollout_completions(
                trainer, prompts=[messages], as_chat=True
            )[0]

        prompt_ids = rollout_out.get("prompt_ids", [])
        completion_ids = rollout_out.get("completion_ids", [])
        logprobs = rollout_out.get("logprobs", [])
        completion_text = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()
        
        messages.append({"role": "assistant", "content": completion_text})

        # Parse action from model output
        action_to_send = remove_reasoning_tags(completion_text)
        if action_to_send.endswith("</s>"):
            action_to_send = action_to_send[:-5]
        if "Action:" in action_to_send:
            action_to_send = action_to_send.split("Action:")[-1].strip()

        strategy_followed = False  # No simple strategy for complex games

        # Step environment with model's action
        invalid_action = False
        
        invalid_action = False
        try:
            action_id_parsed = int(action_to_send.strip())
            hand_cards = get_hand_cards(formatted_observation)
            if action_id_parsed not in hand_cards:
                print(f"Invalid action: {action_id_parsed} not in hand cards: {hand_cards}")
                invalid_action = True
        except Exception:
            invalid_action = True
            print(f"Invalid action: {action_to_send}")
            
        if invalid_action:
            print(f"Messages: {messages}")
            reward = INVALID_PENALTY
        elif strategy_followed:
            # Calculate scale reward for response length, longer responses get lower reward
            response_length = len(completion_ids)
            prompt_length = len(prompt_ids)
            len_reward_scale = max(0.2, min(5, prompt_length / response_length))
            reward = STRATEGY_REWARD * len_reward_scale
        else:
            reward = 0.0
            
        print("--------------------------------")
        print(
            f"[GT] game={game_id} train_turn={target_training_turn} "
            f"strategy={strategy_followed} "
            f"reward={reward:.3f} hints={use_hints}"
        )
        print("--------------------------------")
        
        return index, {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "reward": reward,
            "strategy_followed": strategy_followed,
        }

    # Execute episodes in parallel
    results = [None] * len(prompts)
    executor = rollout_last_prompt_and_completion_parallelized_curriculum.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            # Fallback for failed / short-circuited episodes
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "logprobs": [1.0],
                "reward": 0.0,
                "strategy_followed": False,
            }

    # Update curriculum
    curriculum.step(len(prompts))

    # Update curriculum with per-episode results (performance gating + failure buffer)
    for r in results:
        if r is not None:
            task_id = int(prompts[results.index(r)]) if prompts else 0
            success = r["reward"] > 0
            curriculum.update(task_id, r["reward"], success)

    # Log batch stats
    valid = [r for r in results if r is not None]
    if valid:
        avg_strat = sum(1 for r in valid if r["strategy_followed"]) / len(valid)
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        status = curriculum.get_status()
        print(
            f"[GT-BATCH] Strategy: {avg_strat:.1%}, Avg Reward: {avg_reward:.3f}, "
            f"Win Rate: {status['win_rate']:.1%}, Stage: {status['gated_stage']}/{status['time_stage']}, "
            f"Failure Buffer: {status['failure_buffer_size']}"
        )

    return {
        "prompt_ids": [r["prompt_ids"] for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "logprobs": [r["logprobs"] for r in results],
        "env_rewards": [r["reward"] for r in results],
    }


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for game environments.
    Uses full prompt and completion IDs with action masking.
    """
    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 4225      # Max prompt tokens before ending episode early
    
    # --- Reward Shaping Parameters ---
    STRATEGY_REWARD_WEIGHT = 0.5  # Weight for strategy adherence vs final score
    STEP_STRATEGY_REWARD = 0.1    # Immediate reward for following strategy at each step

    games_to_task_id_range = GAMES_TO_TASK_ID_RANGE

    selected_game = getattr(trainer.args, "selected_game", "gin_rummy")

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_full_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []  # list of dicts: {base_url}

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                # Initialize with a test reset to ensure server is ready
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
                res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
                res.raise_for_status()
                env_pool.append({"base_url": base_url})
                print(f"[INIT] Server {idx} ready")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        rollout_full_prompt_and_completion_parallelized_curriculum.rank = rank
        rollout_full_prompt_and_completion_parallelized_curriculum.env_pool = env_pool
        rollout_full_prompt_and_completion_parallelized_curriculum.num_servers = len(env_pool)
        rollout_full_prompt_and_completion_parallelized_curriculum.initialized = True
        rollout_full_prompt_and_completion_parallelized_curriculum.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        rollout_full_prompt_and_completion_parallelized_curriculum.generation_semaphore = Semaphore(1)
        rollout_full_prompt_and_completion_parallelized_curriculum.games_to_task_id_range = games_to_task_id_range
        rollout_full_prompt_and_completion_parallelized_curriculum.selected_game = selected_game
        
        # Initialize curriculum scheduler
        rollout_full_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=45,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.45,
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=50, rollouts_per_stage={trainer.args.rollouts_per_stage}, initial_hint_prob=0.50, final_hint_prob=0.0, warmup_rollouts={trainer.args.rollouts_per_stage}")

    # Retrieve static variables
    rank = rollout_full_prompt_and_completion_parallelized_curriculum.rank
    env_pool = rollout_full_prompt_and_completion_parallelized_curriculum.env_pool
    num_servers = rollout_full_prompt_and_completion_parallelized_curriculum.num_servers
    games_to_task_id_range = rollout_full_prompt_and_completion_parallelized_curriculum.games_to_task_id_range
    selected_game = rollout_full_prompt_and_completion_parallelized_curriculum.selected_game
    curriculum = rollout_full_prompt_and_completion_parallelized_curriculum.curriculum
    
    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    
    # Get current curriculum parameters
    total_rollouts = curriculum.total_rollouts
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    def run_single_prompt(index: int, prompt: str):
        # Generate a random game_id for this episode
        game_id = int(prompt)

        # Select server based on index and rank
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []
        prev_full_ids: list[int] | None = None
        invalid_count = 0
        done = False
        train_reward = 0.0
        turn_number = 0
        
        # Track strategy adherence
        strategy_followed_count = 0
        total_strategy_opportunities = 0
        step_rewards = [] 
        all_steps_correct = True
        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            # Get episode id for rest of interactions
            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            raw_observation = result_block.get("observation", "")
            formatted_observation = extract_and_format_observation(raw_observation)

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # First make system prompt
        system_prompt = GAME_SYSTEM_PROMPTS.get(selected_game, DEFAULT_SYSTEM_PROMPT)

        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = GAME_STRATEGY_HINTS.get(selected_game, DEFAULT_STRATEGY_HINTS)
            system_prompt += suggestion_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_observation}]

        # --- Interaction Loop ---
        while not done and (turn_number < current_max_turn):
            
            # Generate Rollout Completion
            # Only allow one thread to generate rollout completions at a time
            with rollout_full_prompt_and_completion_parallelized_curriculum.generation_semaphore:
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
                    delta_prompt_ids = prompt_ids[len(prev_full_ids):]
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

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            # --- Step Environment (POST /step) ---
            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data
                raw_observation = step_block.get("observation", "")
                formatted_observation = extract_and_format_observation(raw_observation)
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

            except Exception as e:
                print(f"Step failed: {e}")
                step_reward = -0.01
                done = False
                invalid_count += 1

            # Check for invalid actions in observation
            if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
                invalid_count += 1
                step_rewards.append(-0.05)  # Penalty for invalid action
            else:
                step_rewards.append(0.02)   # Small reward for valid action

            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]
            
        # --- Calculate Final Reward with Strategy Shaping ---
        strategy_ratio = 0.0
        
        # Combine immediate step rewards
        immediate_rewards = sum(step_rewards)
        
        # If game didn't finish (no terminal reward), give partial credit for progress
        if not done and train_reward == 0.0:
            # Partial reward based on how far we got without invalid actions
            valid_ratio = max(0, turn_number - invalid_count) / max(turn_number, 1)
            train_reward = 0.1 * valid_ratio  # Small positive for playing validly
        
        # Use final game score as the primary reward
        shaped_reward = train_reward + immediate_rewards
        
        # Apply invalid action penalty
        shaped_reward = shaped_reward - 0.05 * float(invalid_count)

        # Log
        print(f"[FULL] id={game_id}, turns={turn_number}/{current_max_turn}, invalids={invalid_count}, done={done}, score={train_reward:.3f}, reward={shaped_reward:.3f}")

        return index, {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask": episode_action_mask,
            "logprobs": episode_logprobs,
            "reward": shaped_reward,
            "strategy_ratio": strategy_ratio,
            "final_score": train_reward,
        }

    # --- Execute in parallel ---
    results = [None] * len(prompts)
    executor = rollout_full_prompt_and_completion_parallelized_curriculum.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            # Fallback for failed episodes
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "action_mask": [0],
                "logprobs": [1.0],
                "reward": 0.0,
                "strategy_ratio": 0.0,
                "final_score": 0.0,
            }
            
    # Update curriculum after batch
    curriculum.step(len(prompts))

    # Update curriculum with per-episode results (performance gating + failure buffer)
    for i, r in enumerate(results):
        if r is not None and r.get("final_score") is not None:
            task_id = int(prompts[i]) if i < len(prompts) else 0
            success = r["final_score"] > 0
            curriculum.update(task_id, r["final_score"], success)

    list_results = [r for r in results if r is not None]
    
    # Log batch statistics with curriculum info
    avg_strategy = sum(r["strategy_ratio"] for r in list_results) / len(list_results) if list_results else 0
    avg_final = sum(r["final_score"] for r in list_results) / len(list_results) if list_results else 0
    status = curriculum.get_status()
    print(
        f"[BATCH] Avg Strategy: {avg_strategy:.2%}, Avg Score: {avg_final:.3f}, "
        f"Win Rate: {status['win_rate']:.1%}, Stage: {status['gated_stage']}/{status['time_stage']}, "
        f"Failure Buffer: {status['failure_buffer_size']}"
    )

    # ---- Aggregate ----
    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "action_mask": [r["action_mask"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)