from agent.llm.prompt import build_prompt
from agent.profile.schema import UserProfile

profile = UserProfile(
    name="Henrique",
    interests=["ESO", "CV", "Robotics"],
    goals=["become an AI engineer"],
)
prompt = build_prompt("What should I study next?", profile=profile)
print(prompt)
