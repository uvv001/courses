from smolagents import CodeAgent, DuckDuckGoSearchTool
from arguments_helper import ArgumentsHelper

def main():
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=ArgumentsHelper().getModel())
    agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

if __name__ == "__main__":
    main()