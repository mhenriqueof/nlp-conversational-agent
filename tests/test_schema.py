from agent.profile.schema import UserProfile

profile = UserProfile(
    name="Henrique",
    interests=["ESO", "AI"],
    goals=["become an AI engineer"],
    values=["empathy"],
)
print(profile.to_prompt_string())
