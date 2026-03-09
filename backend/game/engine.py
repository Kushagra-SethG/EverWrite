from llm.gemini import generate_response, generate_response_stream
from memory.vector_store import add_memory, get_memory
from game.prompt import build_prompt
from game.state import GameState


def detect_faction(text):
    factions = [
        "Lumina Concordat",
        "Cogwheel Dominion",
        "Verdant Enclave",
        "Obsidian Vault",
        "Ashfall Clans"
    ]

    for f in factions:
        if f.lower() in text.lower():
            return f
    return None


def process_turn_stream(user_input, state):
    """
    Generator used by the web API.
    Yields text chunks from the LLM, then updates state and memory.
    Yields (chunk, is_done, state_dict) tuples.
    is_done is only True on the final sentinel yield.
    """
    memory = get_memory(user_input)
    prompt = build_prompt(user_input, memory, state)

    full_response = ""
    for chunk in generate_response_stream(prompt):
        full_response += chunk
        yield chunk, False, {}

    # --- state transitions after full response ---
    if state.phase == "intro":
        faction = detect_faction(user_input)
        if faction:
            state.update_faction(faction)

    elif state.phase == "equipment":
        state.update_equipment(user_input)

    if "THE JOURNEY ENDS" in full_response:
        state.end_game()

    add_memory(f"User: {user_input}")
    add_memory(f"AI: {full_response}")

    yield "", True, {
        "phase": state.phase,
        "faction": state.faction,
        "equipment": state.equipment,
    }


def run_game():
    state = GameState()

    print("Game Started (type exit to quit)\n")

    while True:
        user_input = input("> ")

        if user_input.lower() == "exit":
            break

        memory = get_memory(user_input)

        prompt = build_prompt(user_input, memory, state)

        response = generate_response(prompt)

        print("\n" + response + "\n")

        if state.phase == "intro":
            faction = detect_faction(user_input)
            if faction:
                state.update_faction(faction)

        elif state.phase == "equipment":
            state.update_equipment(user_input)

        add_memory(f"User: {user_input}")
        add_memory(f"AI: {response}")

        if "THE JOURNEY ENDS" in response:
            state.end_game()
            break