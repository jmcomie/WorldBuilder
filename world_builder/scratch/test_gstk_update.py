from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_experimental.smart_llm import SmartLLMChain

from dotenv import load_dotenv

load_dotenv()

prompt_str: str = """You are creating a 2D tilemap for a 2D game represented as integers in a CSV.
Create a eight by eight CSV file's contents in which each cell is one of the
gid integers below, and that adheres to the following description: Create a map of the beach on the eastern side of some geography where the shore is aligned vertically near the center of the map and in which grass is on the left side of the map and in which water is on the right side of the map, with an appropriate gradation.
GID Descriptions:

gid: 1 description: light green grass
gid: 2 description: light green grass accented with leaves arranged from lower left to upper right
gid: 3 description: light green grass accented with leaves arranged from upper left to lower right
gid: 4 description: green grass
gid: 5 description: green grass accented with leaves arranged from lower left to upper right
gid: 6 description: green grass accented with leaves arranged from upper left to lower right
gid: 7 description: sand
gid: 8 description: ankle deep water
gid: 9 description: knee deep water
gid: 10 description: shoulder deep water
gid: 11 description: water too deep to stand in
"""


prompt = PromptTemplate.from_template(prompt_str)
chain = SmartLLMChain(
    ideation_llm=ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    critique_llm=ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    resolver_llm=ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    prompt=prompt,
    n_ideas=3,
    verbose=True,
)
# Need to add arguments to run.
response = chain.run({})
print(response)