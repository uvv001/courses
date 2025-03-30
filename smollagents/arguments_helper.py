import os
from smolagents import HfApiModel, LiteLLMModel

class ArgumentsHelper:
    def __init__(self):
        import argparse
        parser = argparse.ArgumentParser(description="Run the CodeAgent with a specified provider.")
        parser.add_argument("--provider", choices=["hg", "openrouter"], default="hg", help="Model provider: 'hg' (default) or 'openrouter'.")
        self.args = parser.parse_args()

    def getModel(self):
        if self.args.provider == "hg":
            return HfApiModel()
        
        if self.args.provider == "openrouter":
            return LiteLLMModel(
                model_id="openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
                api_key=os.environ["OPENROUTER_API_KEY"]
            )
