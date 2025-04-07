import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
import numexpr
import re
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text Any Math Problem", page_icon="👾")
st.title("Text any problem I will solve the Puzzle 👾 ")

groq_api_key = st.sidebar.text_input(label="Groq Api Key", type="password")

if not groq_api_key:
    st.info("Please enter a valid Groq API key.")
    st.stop()

llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching math-related problems and concepts.",
)

def calculate_expression(expression):
    try:
        # Clean the expression - remove markdown formatting and extra text
        if "```text" in expression:
            expression = expression.split("```text")[1].split("```")[0].strip()
        elif "```" in expression:
            expression = expression.split("```")[1].split("```")[0].strip()
        
        # Extract just the math expression using regex
        math_expr = re.search(r"(\d+[\s\*\+\-\/\^\.]\d+)", expression)
        if math_expr:
            expression = math_expr.group(1)
        
        # Remove any whitespace from the expression
        expression = expression.replace(" ", "")
        
        # Evaluate safely
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

calculator_tool = Tool(
    name="calculator",
    func=calculate_expression,
    description="Useful for performing mathematical calculations. Input should be a plain mathematical expression like '2+2' or '37593*67'."
)
prompt_template = """
Your agent is tasked with solving math-related problems and questions. Analyze the question carefully and provide a logical, step-by-step answer. Present the solution clearly, numbering each step.

Question: {question}
Answer:
"""

reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template
)

reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_chain.run,
    description="A tool for providing detailed, step-by-step solutions to math problems.",
)

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot ready to solve your puzzles! 👾"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    

if prompt := st.chat_input("Enter your math question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Thinking..."):
        streamlit_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response = assistant_agent.run(prompt, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")