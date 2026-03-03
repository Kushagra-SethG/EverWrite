class GameState:
    def __init__(self):
        self.phase = "intro"
        self.faction = None
        self.equipment = None

    def update_faction(self, faction):
        self.faction = faction
        self.phase = "equipment"

    def update_equipment(self, equipment):
        self.equipment = equipment
        self.phase = "story"

    def end_game(self):
        self.phase = "ended"