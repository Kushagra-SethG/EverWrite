import sys
import os

# Resolve sibling packages (game, llm, memory, config) regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from game.engine import run_game

if __name__ == "__main__":
    run_game()