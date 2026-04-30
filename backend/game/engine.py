import json
import re
from llm.gemini import generate_response, generate_response_stream
from memory.vector_store import add_memory, get_memory
from game.prompt import build_prompt
from game.state import GameState


FACTION_ALIASES = {
    "Lumina Concordat": [
        "lumina concordat", "lumina", "concordat", "light guardians", "light guardian", "healer", "healers",
    ],
    "Cogwheel Dominion": [
        "cogwheel dominion", "cogwheel", "dominion", "innovators", "engineers", "engineer", "technology", "tech",
    ],
    "Verdant Enclave": [
        "verdant enclave", "verdant", "enclave", "nature keepers", "nature keeper", "druids", "druid", "nature",
    ],
    "Obsidian Vault": [
        "obsidian vault", "obsidian", "vault", "shadow", "shadowy practitioners", "dark magic", "forbidden",
    ],
    "Ashfall Clans": [
        "ashfall clans", "ashfall", "warriors", "warrior", "warlord", "volcanic", "berserker",
    ],
}


def _normalize_text(text):
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())


def detect_faction(text):
    normalized = _normalize_text(text)

    # Support numeric faction picks, e.g. "I choose 5" or "the second one".
    numeric_map = {
        "1": "Lumina Concordat",
        "2": "Cogwheel Dominion",
        "3": "Verdant Enclave",
        "4": "Obsidian Vault",
        "5": "Ashfall Clans",
        "first": "Lumina Concordat",
        "second": "Cogwheel Dominion",
        "third": "Verdant Enclave",
        "fourth": "Obsidian Vault",
        "fifth": "Ashfall Clans",
    }
    for token, faction in numeric_map.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            return faction

    # Exact alias/keyword matching for flexible player phrasing.
    for faction, aliases in FACTION_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return faction

    # Heuristic fallback for broad banner phrasing (e.g. "I choose the warrior banner").
    heuristic_keywords = {
        "Lumina Concordat": {"light", "heal", "healing", "restore", "restoration"},
        "Cogwheel Dominion": {"machine", "mechanical", "gear", "gadget", "craft", "invention"},
        "Verdant Enclave": {"forest", "grove", "beast", "herb", "green", "balance"},
        "Obsidian Vault": {"shadow", "dark", "secret", "forbidden", "stealth"},
        "Ashfall Clans": {"battle", "combat", "strength", "honor", "fire", "war"},
    }
    words = set(normalized.split())
    best_faction = None
    best_score = 0
    for faction, keys in heuristic_keywords.items():
        score = len(words.intersection(keys))
        if score > best_score:
            best_score = score
            best_faction = faction
    if best_score >= 2:
        return best_faction

    return None


def extract_name(user_input):
    """
    Extract a name from natural language input.
    Handles patterns like:
    - "my name is arther"
    - "i'm arther"
    - "i am arther"
    - "call me arther"
    - "you can call me arther"
    - "just call me arther"
    - "the name is arther"
    - Plain name: "arther"
    """
    text = user_input.strip().lower()
    
    # Pattern: "my name is [name]"
    match = re.search(r"my name is\s+([a-z'\s-]+)", text)
    if match:
        name = match.group(1).strip().title()
        if len(name) > 1:
            return name[:30]
    
    # Pattern: "i'm [name]" or "i am [name]"
    match = re.search(r"(?:i'm|i am)\s+([a-z'\s-]+?)(?:\.|,|$)", text)
    if match:
        name = match.group(1).strip().title()
        if len(name) > 1:
            return name[:30]
    
    # Pattern: "call me [name]" or "you can call me [name]" or "just call me [name]"
    match = re.search(r"(?:you\s+)?(?:can|may|just)?\s*call me\s+([a-z'\s-]+?)(?:\.|,|if|$)", text)
    if match:
        name = match.group(1).strip().title()
        if len(name) > 1:
            return name[:30]
    
    # Simple pattern: "[you can] call me [name]"
    match = re.search(r"call me\s+([a-z'\s-]+)(?:\.|,|$)", text)
    if match:
        name = match.group(1).strip().title()
        if len(name) > 1:
            return name[:30]
    
    # Pattern: "the name is [name]" or "name is [name]"
    match = re.search(r"(?:the\s+)?name is\s+([a-z'\s-]+)", text)
    if match:
        name = match.group(1).strip().title()
        if len(name) > 1:
            return name[:30]
    
    # Pattern: "my name's [name]"
    match = re.search(r"my name'?s?\s+([a-z'\s-]+)", text)
    if match:
        name = match.group(1).strip().title()
        if len(name) > 1:
            return name[:30]
    
    # Fallback: take first word-like sequence if no pattern matches
    words = [w for w in text.split() if w.isalpha() and len(w) > 1]
    if words:
        # Skip common words
        skip_words = {'you', 'can', 'call', 'me', 'just', 'the', 'my', 'name', 'is', 'am', 'i', 'are', 'be'}
        for word in words:
            if word not in skip_words:
                return word.title()
    
    # Last resort: use the whole input, formatted
    return user_input.strip()[:30].title()


def _parse_int_value(value, default=0):
    match = re.search(r"[-+]?\d+", value or "")
    if not match:
        return default
    try:
        return int(match.group(0))
    except ValueError:
        return default


def parse_consequence_block(response_text):
    """Extract [CONSEQUENCE]...[/CONSEQUENCE] block and parse it."""
    pattern = r"\[CONSEQUENCE\](.*?)\[/CONSEQUENCE\]"
    match = re.search(pattern, response_text, re.DOTALL)
    if not match:
        return {}
    
    block = match.group(1).strip()
    consequence = {}
    
    for line in block.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        
        if key == "health_change":
            consequence["health_change"] = _parse_int_value(value)
        elif key == "attunement_change":
            consequence["attunement_change"] = _parse_int_value(value)
        elif key == "influence_change":
            consequence["influence_change"] = _parse_int_value(value)
        elif key == "knowledge_change":
            consequence["knowledge_change"] = _parse_int_value(value)
        elif key == "item_gained":
            consequence["item_gained"] = value if value and value.lower() != "null" else None
        elif key == "item_lost":
            consequence["item_lost"] = value if value and value.lower() != "null" else None
        elif key == "event":
            consequence["event"] = value if value and value.lower() != "null" else None
        elif key == "location":
            consequence["location"] = value if value and value.lower() != "null" else None
        elif key == "npc_met":
            consequence["npc_met"] = value if value and value.lower() != "null" else None
        elif key == "faction_changed":
            if value and value.lower() != "{}":
                try:
                    consequence["faction_changed"] = json.loads(value)
                except:
                    pass
    
    return consequence


def strip_consequence_block(response_text):
    """Remove [CONSEQUENCE] block from narrative text shown to player."""
    return re.sub(r"\[CONSEQUENCE\].*?\[/CONSEQUENCE\]", "", response_text, flags=re.DOTALL).strip()


def process_turn_stream(user_input, state):
    """Stream AI response, then extract and apply state changes."""

    # Apply phase transitions before prompting so the current turn uses the
    # correct phase guidance instead of responding one step behind.
    if state.phase == "name" and user_input.strip():
        state.character_name = extract_name(user_input)
        state.phase = "intro"
    elif state.phase == "intro" and state.faction is not None:
        state.phase = "equipment"
    
    memory = get_memory(user_input)
    prompt = build_prompt(user_input, memory, state)

    full_response = ""
    visible_emitted = 0
    marker = "[CONSEQUENCE]"

    for chunk in generate_response_stream(prompt):
        full_response += chunk

        marker_idx = full_response.find(marker)
        if marker_idx != -1:
            visible_target = full_response[:marker_idx]
        else:
            # Keep a small suffix buffered so partial marker fragments are not leaked.
            safe_len = max(0, len(full_response) - len(marker) + 1)
            visible_target = full_response[:safe_len]

        if len(visible_target) > visible_emitted:
            delta = visible_target[visible_emitted:]
            visible_emitted = len(visible_target)
            yield delta, False, {}

    # Flush any remaining visible text when there is no consequence block.
    if marker not in full_response and len(full_response) > visible_emitted:
        delta = full_response[visible_emitted:]
        if delta:
            yield delta, False, {}

    narrative_response = strip_consequence_block(full_response)

    # Parse consequences from full raw response
    consequence = parse_consequence_block(full_response)
    if consequence:
        state.apply_consequence(consequence)

    # Handle phase transitions
    if state.phase == "intro":
        # Faction detection: player chose a faction
        faction = detect_faction(user_input)
        if faction:
            state.update_faction(faction)

    elif state.phase == "equipment":
        # Equipment phase accepts any input
        if user_input.strip():
            state.update_equipment(user_input.strip()[:50])

    # Check for game end
    if "THE JOURNEY ENDS" in narrative_response or state.health <= 0:
        state.end_game()

    # Store memory (without raw consequence metadata)
    add_memory(f"User: {user_input}")
    add_memory(f"AI: {narrative_response}")

    yield "", True, state.to_dict()


def run_game():
    state = GameState()
    print("Game Started (type exit to quit)\\n")

    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break

        memory = get_memory(user_input)
        prompt = build_prompt(user_input, memory, state)
        response = generate_response(prompt)
        narrative_response = strip_consequence_block(response)
        print("\\n" + narrative_response + "\\n")

        consequence = parse_consequence_block(response)
        if consequence:
            state.apply_consequence(consequence)

        # Handle phase transitions
        if state.phase == "name":
            if user_input.strip():
                state.character_name = user_input.strip()[:30]
                state.phase = "intro"
        elif state.phase == "intro":
            faction = detect_faction(user_input)
            if faction:
                state.update_faction(faction)
        elif state.phase == "equipment":
            if user_input.strip():
                state.update_equipment(user_input.strip()[:50])

        add_memory(f"User: {user_input}")
        add_memory(f"AI: {narrative_response}")

        if "THE JOURNEY ENDS" in narrative_response or state.health <= 0:
            state.end_game()
            break

