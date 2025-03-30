from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

def main():
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
    agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")


if __name__ == "__main__":
    main()