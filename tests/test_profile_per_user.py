from agent.profile.schema import UserProfile

profile = UserProfile(name="Henrique", interests=["ESO", "AI"])
profile.save(username="henrique2026")

loaded = UserProfile.load(username="henrique2026")
print(loaded.to_prompt_string())
