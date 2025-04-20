from smolagents import ToolCallingAgent, DuckDuckGoSearchTool
from helpers import ArgumentsHelper, register_opentelemetry_through_langfuse

register_opentelemetry_through_langfuse()

def main():
    agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=ArgumentsHelper().getModel(), verbosity_level=2)
    agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

if __name__ == "__main__":
    main()