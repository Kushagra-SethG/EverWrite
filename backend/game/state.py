import json
from typing import Optional


class GameState:
    def __init__(self):
        self.phase = "name"  # Start by asking for character name
        self.character_name = None
        self.reincarnated_origin = "Earth"
        self.world_name = "Aethel"
        self.clan = None
        self.faction = None
        self.equipment = None
        self.health = 10
        self.aetheric_attunement = 0
        self.influence = 0
        self.knowledge = 0
        self.inventory = []
        self.relationships = {"Lumina Concordat": 0, "Cogwheel Dominion": 0, "Verdant Enclave": 0, "Obsidian Vault": 0, "Ashfall Clans": 0}
        self.events = []
        self.locations_visited = []
        self.npcs_met = []

    def update_faction(self, faction):
        self.faction = faction
        self.phase = "equipment"
        self.relationships[faction] += 5
        self.events.append(f"Joined {faction}")

    def update_equipment(self, equipment):
        self.equipment = equipment
        self.phase = "story"
        self.inventory.append(equipment)
        self.events.append(f"Acquired {equipment}")

    def apply_consequence(self, c):
        if "health_change" in c:
            self.health = max(0, min(10, self.health + c["health_change"]))
            if self.health <= 0: self.end_game()
        if "attunement_change" in c:
            self.aetheric_attunement = max(0, min(10, self.aetheric_attunement + c["attunement_change"]))
        if "influence_change" in c:
            self.influence = max(0, min(10, self.influence + c["influence_change"]))
        if "knowledge_change" in c:
            self.knowledge = max(0, min(10, self.knowledge + c["knowledge_change"]))
        if "item_gained" in c: self.inventory.append(c["item_gained"])
        if "item_lost" in c and c["item_lost"] in self.inventory: self.inventory.remove(c["item_lost"])
        if "faction_changed" in c:
            for faction, change in c["faction_changed"].items():
                if faction in self.relationships:
                    self.relationships[faction] = max(-10, min(10, self.relationships[faction] + change))
        if "event" in c: self.events.append(c["event"])
        if "location" in c and c["location"] not in self.locations_visited:
            self.locations_visited.append(c["location"])
        if "npc_met" in c and c["npc_met"] not in self.npcs_met:
            self.npcs_met.append(c["npc_met"])

    def end_game(self):
        self.phase = "ended"

    def to_dict(self):
        return {
            "phase": self.phase, "character_name": self.character_name, "reincarnated_origin": self.reincarnated_origin, "world_name": self.world_name, "clan": self.clan, "faction": self.faction, "equipment": self.equipment,
            "health": self.health, "attunement": self.aetheric_attunement,
            "influence": self.influence, "knowledge": self.knowledge,
            "inventory": self.inventory, "relationships": self.relationships,
            "locations": self.locations_visited, "npcs": self.npcs_met,
            "events_count": len(self.events),
        }

    def to_state_summary(self):
        inv_str = ", ".join(self.inventory) if self.inventory else "nothing"
        locs_str = ", ".join(self.locations_visited[-3:]) if self.locations_visited else "none yet"
        npcs_str = ", ".join(self.npcs_met[-3:]) if self.npcs_met else "none yet"
        events_str = ", ".join(self.events[-3:]) if self.events else "none yet"
        rel = "`n".join(f"  {n}: {v:+d}" for n, v in self.relationships.items())
        return f"REINCARNATION PREMISE:`n- Origin World: {self.reincarnated_origin}`n- New World: {self.world_name}`n- Birth Status: Reincarnated soul in a clan-based world`n`nCHARACTER STATUS:`n- Name: {self.character_name or 'Unnamed'}`n- Clan/Faction: {self.faction or self.clan or 'None'}`n- Health: {self.health}/10`n- Attunement: {self.aetheric_attunement}/10`n- Influence: {self.influence}/10`n- Knowledge: {self.knowledge}/10`n- Inventory: {inv_str}`n`nWORLD STATE:`n- Equipment: {self.equipment or 'None'}`n- Locations: {locs_str}`n- NPCs: {npcs_str}`n- Recent Events: {events_str}`n- Events: {len(self.events)}`n`nFACTION RELATIONS:`n{rel}"
