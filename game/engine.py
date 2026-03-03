from llm.gemini import generate_response
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