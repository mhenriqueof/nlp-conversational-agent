from agent.profile.schema import UserProfile
from agent.profile.updater import ProfileUpdater

profile = UserProfile()
updater = ProfileUpdater()

profile = updater.extract_and_update(
    "Hi! My name is Henrique and I like AI. My goal is to become an AI engineer.",
    profile,
)
print(profile.to_prompt_string())
