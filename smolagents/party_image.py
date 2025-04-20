from smolagents import CodeAgent, Tool
from helpers import ArgumentsHelper, register_opentelemetry_through_langfuse

register_opentelemetry_through_langfuse()

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

agent = CodeAgent(tools=[image_generation_tool], model=ArgumentsHelper().getModel())

agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={'user_prompt': 'A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala'}
)