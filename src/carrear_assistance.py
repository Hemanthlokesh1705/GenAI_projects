from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_classic.chains.sequential import SequentialChain
from dotenv import load_dotenv
load_dotenv()
first_template="""
            ROLE: Act as Placement Officer with 10+ years of experience.

            Task: Answer the student question: {question}
            Goal: Help them crack better placement opportunities.
            Rules:
            - Be aligned with current hiring trends
            - Avoid outdated and generic advice
            - Be practical and actionable

            Output: Career advice
            """
example_prompt1 = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}\n"
)

second_template="""
    ROLE:Act as an motivational speaker
    Analyze:{prompt1_output}
    Goal:
    Give 2-3 lines of motivation.
    make student feel confident.
    Tell 1 fact about life
"""
examples = [
    {
        "question": "I am weak in coding",
        "answer": "Start with basics, practice daily and build projects."
    },
    {
        "question": "I fear interviews",
        "answer": "Mock interviews and communication practice will boost confidence."
    }
]


few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt1,
    prefix="You are a placement officer. Answer student questions.\n",
    suffix="Question: {question}\nAnswer:",
    input_variables=["question"]
)

prompt2=PromptTemplate(
    input_variables=["prompt1_output"],
    template=second_template
)
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    max_tokens=100
)
chain1 = LLMChain(llm=llm, prompt=few_shot_prompt, output_key="prompt1_output")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="final_output")
final_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["question"],
    output_variables=["prompt1_output", "final_output"]
)

user_prompt=input("Hello,How can i help you..?")
if user_prompt:
    print(final_chain({"question":user_prompt}))