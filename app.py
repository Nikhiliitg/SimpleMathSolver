import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set up the Streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🪝")
st.title("Text to Math Problem Solver using Groq's Gemma 2")

groq_api_key = st.sidebar.text_input(label="Enter Your Groq API Key", type="password")

# Check for API key
if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# Initialize the Groq model
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

## Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching Wikipedia for various information on the topic mentioned."
)

## Initializing the math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only a mathematical expression needs to be provided."
)

# Adjusted prompt template to avoid forbidden characters and invalid code
prompt = """
You are tasked with solving mathematical problems. Please return only a valid mathematical expression that can be evaluated using basic Python math operations. Avoid imports or any additional code. Return the expression in its simplest form.
Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template=prompt
)

## Combine tools to chain
reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## Initializing the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a Math Chatbot who can answer all your math-related queries."}
    ]

# Display messages from the session state
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# User input for the question
question = st.text_area("Enter Your Question:")

## Start the interaction
if st.button("Find my answer"):
    if question.strip():  # Validate non-empty input
        with st.spinner("Generating response...."):
            # Add user question to session messages
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            
            # Generate the response from the agent
            st_cb = StreamlitCallbackHandler(st.container())
            try:
                response = assistant_agent.run(question, callbacks=[st_cb])

                # Ensure the response contains only a valid expression
                sanitized_response = response.strip()
                
                # Attempt to evaluate the mathematical expression
                try:
                    result = eval(sanitized_response)
                    st.session_state.messages.append({'role': 'assistant', 'content': f"{sanitized_response} = {result}"})
                    st.write("## Response:")
                    st.success(f"{sanitized_response} = {result}")
                except Exception as eval_error:
                    st.error(f"Error evaluating expression: {sanitized_response}. Details: {eval_error}")
                    st.session_state.messages.append({'role': 'assistant', 'content': f"Error in evaluation: {sanitized_response}"})
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")

