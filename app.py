import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# ============ Initialize Models ============
def get_debate_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)

def get_judge_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# ============ Debate Prompt ============
debate_template = """
You are an expert debator who have won multiple debate contests. The user will pick a topic and a stance (Pro or Con).
You must always take the OPPOSITE stance and debate with them. Keep the size of your argument similar to that of the user.

Topic: {topic}
User's stance: {user_stance}
AI's stance: Opposite of the user

Respond with strong, structured counter-arguments.
"""

# ============ Judge Prompt ============
judge_template = """
You are the debate judge.

Here is the complete debate transcript:
{debate_transcript}

The user had stance: {user_stance}
The AI had the opposite stance.

Judge fairly based on:
- Strength of arguments
- Relevance to the topic
- Logical reasoning

Declare the WINNER: either "User" or "AI".
Also give a 3–4 sentence explanation of why the winner is the winner.
"""

# ============ Streamlit UI ============
st.set_page_config(page_title="Project Debator", layout="wide")
st.title("Project Debator")
st.write("Debate against an AI model. Choose your topic, stance, and number of rounds.")

# Sidebar Inputs
with st.sidebar:
    topic = st.text_input("Enter debate topic:")
    stance = st.selectbox("Choose your stance:", ["Pro", "Con"])
    rounds = st.number_input("Number of rounds:", min_value=1, max_value=10, value=3)
    start_btn = st.button("Start Debate")

# Session state for conversation
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "current_round" not in st.session_state:
    st.session_state.current_round = 1
if "debate_started" not in st.session_state:
    st.session_state.debate_started = False
if "ai_responses" not in st.session_state:
    st.session_state.ai_responses = []
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []

# Start Debate
if start_btn and topic:
    st.session_state.debate_started = True
    st.session_state.current_round = 1
    st.session_state.memory.clear()
    st.session_state.ai_responses = []
    st.session_state.user_inputs = []
    st.success(f"Debate started on {topic} | Your stance: {stance} | AI stance: {'Pro' if stance=='Con' else 'Con'}")

# Debate Rounds
if st.session_state.debate_started and st.session_state.current_round <= rounds:
    st.subheader(f"Round {st.session_state.current_round} / {rounds}")

    user_argument = st.text_area("Your Argument:", key=f"round_{st.session_state.current_round}")
    if st.button("Submit Argument", key=f"submit_{st.session_state.current_round}"):
        if user_argument.strip():
            # Store user input
            st.session_state.user_inputs.append(user_argument)

            # Debate chain
            debate_llm = get_debate_llm()
            debate_chain = ConversationChain(
                llm=debate_llm,
                memory=st.session_state.memory,
                verbose=False
            )

            response = debate_chain.predict(
                input=f"Topic: {topic}\nUser stance: {stance}\nUser says: {user_argument}\nAI responds with the opposite stance."
            )

            st.session_state.ai_responses.append(response)
            st.success(f"AI: {response}")

            # Move to next round
            st.session_state.current_round += 1
        else:
            st.warning("Please enter your argument before submitting.")

# After Debate Ends -> Judge
if st.session_state.debate_started and st.session_state.current_round > rounds:
    st.subheader("Final Judgment")

    # Collect transcript
    transcript = ""
    for i in range(len(st.session_state.user_inputs)):
        transcript += f"Round {i+1}:\nUser: {st.session_state.user_inputs[i]}\nAI: {st.session_state.ai_responses[i]}\n\n"

    # Judge LLM
    judge_llm = get_judge_llm()
    judge_prompt = ChatPromptTemplate.from_template(judge_template)
    judge_input = judge_prompt.format_messages(
        debate_transcript=transcript,
        user_stance=stance
    )
    result = judge_llm(judge_input)

    st.write(result.content)

