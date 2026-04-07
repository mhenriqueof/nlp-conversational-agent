from agent.profile.schema import UserProfile

profile = UserProfile(name="Henrique", interests=["ESO", "AI"])
profile.save()
loaded = UserProfile.load()
print(loaded.to_prompt_string())
