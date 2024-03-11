from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_experimental.smart_llm import SmartLLMChain

from dotenv import load_dotenv

load_dotenv()

prompt_str: str = """SYSTEM MESSAGE:
---------------
The root prompt is a description of the entire map. It is the first prompt in the recursive chain of prompts that describe the map.

You are assisting in the creation of an 81x81 tile map.  To better utilize your internal semantic vector space, rather than being asked to create the 81x81 tile map in one go, you will create a 3x3 p
rompt matrix that divides the entire map into 9 cells. Each cell is to contain a prompt describing its corresponding projected area on the map, with a linguistic complexity appropriate to the map des
cription. From there, you will create a 3x3 prompt matrix representing the areas below each of the 9 initial cells, and so on, until you have created the 81x81 tile map.  Note that, accordingly, each
 level of the map contains nine times more total cells than the level above it, and each cell corresponds to nine times less area of the tile map than the cells in the level above it.

To assist in your creation of the map via these matrices, you are provided the following context: the overall map description, each recursive parent prompt above the matrix being created, and the pos
ition of each recursive parent prompt in the 3x3 matrix in which it was created. For example, a map root description of a circle of trees spanning the entire map might have in position 1,1 of its ini
tial prompt matrix a directive for creating the left middle of the circle of trees (approximating one eighth the arc of the circle), and in position 3,3 of its initial prompt matrix a directive for c
reating the bottom right of the circle of trees, and so on. The recursive parent prompts are provided to help you maintain consistency with the map description as you create the 3x3 prompt matrices.

The following illustrates the structure of the parent context provided to you and highlights the recursive nature of the prompts and their increasing specificity with respect to tile map area:

Map description: A description of the entire map.


USER MESSAGE:
-------------
Create the first 3x3 matrix for the following map description: Draw an island that has some beach, covers about 80% of the map, is 
round but not a perfect circle, and has a small village in the center among trees, with beaches on the edges.


ASSISTANT MESSAGE:
------------------
tiles:
[['Describe the upper-left section of the island, focusing on the irregular coastline and the transition from sea to beach.', 'Describe the upper middle section of the island, focusing on the beach and the start of the forested area.', 'Describe the upper right section of the island, focusing on the irregular coastline and the transition from beach to sea.'], ['Describe the middle left section of the island, focusing on the transition from beach to forest.', 'Describe the center of the island, focusing on the small village nestled among the trees.', 'Describe the middle right section of the island, focusing on the transition from forest to beach.'], ['Describe the lower left section of the island, focusing on the irregular coastline and the transition from beach to sea.', 'Describe the lower middle section of the island, focusing on the beach and the start of the forested area.', 'Describe the lower right section of the island, focusing on the irregular coastline and the transition from sea to beach.']]


SYSTEM MESSAGE:
---------------
The root prompt is a description of the entire map. It is the first prompt in the recursive chain of prompts that describe the map.

You are assisting in the creation of an 81x81 tile map.  To better utilize your internal semantic vector space, rather than being asked to create the 81x81 tile map in one go, you will create a 3x3 prompt matrix that divides the entire map into 9 cells. Each cell is to contain a prompt describing its corresponding projected area on the map, with a linguistic complexity appropriate to the map description. From there, you will create a 3x3 prompt matrix representing the areas below each of the 9 initial cells, and so on, until you have created the 81x81 tile map.  Note that, accordingly, each level of the map contains nine times more total cells than the level above it, and each cell corresponds to nine times less area of the tile map than the cells in the level above it.

To assist in your creation of the map via these matrices, you are provided the following context: the overall map description, each recursive parent prompt above the matrix being created, and the position of each recursive parent prompt in the 3x3 matrix in which it was created. For example, a map root description of a circle of trees spanning the entire map might have in position 1,1 of its initial prompt matrix a directive for creating the left middle of the circle of trees (approximating one eighth the arc of the circle), and in position 3,3 of its initial prompt matrix a directive for creating the bottom right of the circle of trees, and so on. The recursive parent prompts are provided to help you maintain consistency with the map description as you create the 3x3 prompt matrices.

The following illustrates the structure of the parent context provided to you and highlights the recursive nature of the prompts and their increasing specificity with respect to tile map area:

Map description: A description of the entire map.
First level prompt describes subset area one ninth the size of the entire map, at the provided position in the 3x3 inital matrix.


USER MESSAGE:
-------------
Create a second level 3x3 matrix for the following prompt: Describe the upper-left section of the island, focusing on the irregular coastline and the transition from sea to beach.

Parent context:

Map description: Draw an island that has some beach, covers about 80% of the map, is round but not a perfect circle, and has a small village in the center among trees, with beaches on the edges.

First level prompt: Describe the upper-left section of the island, focusing on the irregular coastline and the transition from sea to beach.
"""


prompt = PromptTemplate.from_template(prompt_str)
chain = SmartLLMChain(
    ideation_llm=ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    critique_llm=ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    resolver_llm=ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    prompt=prompt,
    n_ideas=2,
    verbose=True,

)
# Need to add arguments to run.
response = chain.run({})
print(response)