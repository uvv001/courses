from smolagents import CodeAgent, DuckDuckGoSearchTool
from helpers import ArgumentsHelper, register_opentelemetry_through_langfuse

register_opentelemetry_through_langfuse()

def main():
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=ArgumentsHelper().getModel())
    agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

if __name__ == "__main__":
    main()