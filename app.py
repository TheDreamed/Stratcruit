import streamlit as st
import openai
from langchain_openai import ChatOpenAI
import sqlite3
from langchain.prompts import PromptTemplate
import os
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from docx import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import json

# Load the API key from secrets.env
load_dotenv('secrets.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the SQLite database connection
engine = create_engine('sqlite:///resumes.db')

# Create resume table if not exists
def create_resume_db():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY,
        name TEXT,
        experience INTEGER,
        skills TEXT,
        education TEXT,
        achievements TEXT
    )''')
    
    conn.commit()
    conn.close()

# Extract text from PDF files
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Extract text from DOCX files
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract text from uploaded file
def extract_text_from_file(file):
    if file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        return file.read().decode('utf-8')

# Extract structured resume details using the new OpenAI API
# Extract structured resume details using the new OpenAI API
# Extract structured resume details using the new OpenAI API
def extract_resume_details_with_agent(resume_text):
    # Prepare input for the agent
    input_str = f"""
    Please extract the following information from the resume and return it as a well-structured JSON object.
    Ensure that the output is properly formatted and avoid unnecessary comments or explanations.

    Resume:
    {resume_text}

    Expected output format:
    {{
      "name": "Your Name",
      "experience": 0,  // Number of years of experience
      "skills": [
        "Skill 1",
        "Skill 2",
        ...
      ],
      "education": "Your Education",
      "achievements": [
        "Achievement 1",
        "Achievement 2",
        ...
      ]
    }}
    """
    # Initialize the model
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Create an agent with the correct prompt template
    agent = create_react_agent(model, tools, sql_prompt)

    # Create an agent executor and specify the expected output key
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_only_outputs=True)

    # Execute the agent with the provided input and specify the output key
    response = agent_executor.invoke({"input": input_str})
    
    return response.get("output")  # Ensure the output is fetched correctly

    
# Parse the extracted resume details
# Parse the extracted resume details
# Parse the extracted resume details
def parse_extracted_text(extracted_text):
    try:
        # If extracted_text is already a dict, use it directly
        if isinstance(extracted_text, dict):
            extracted_data = extracted_text
        else:
            extracted_data = json.loads(extracted_text)  # If it's a JSON string

        # Check for expected fields
        if 'name' not in extracted_data or 'experience' not in extracted_data or 'skills' not in extracted_data:
            st.error(f"Missing expected fields in resume details: {extracted_text}")
            return {}
        
        name = extracted_data.get("name", "N/A")
        experience = extracted_data.get("experience", 0)
        skills = extracted_data.get("skills", [])
        education = extracted_data.get("education", "N/A")
        achievements = extracted_data.get("achievements", [])

        return {
            "name": name,
            "experience": experience,
            "skills": skills,
            "education": education,
            "achievements": achievements
        }
        
    except json.JSONDecodeError:
        st.error("Failed to parse resume details. The response is not valid JSON.")
        st.write(extracted_text)
        return {}


# Insert structured data into the database
def insert_resume_into_db(parsed_resume_details):
    # Convert skills and achievements to comma-separated strings
    skills_str = ', '.join(parsed_resume_details.get('skills', []))
    achievements_str = ', '.join(parsed_resume_details.get('achievements', []))
    
    # Connect to the database (change 'your_database.db' to your actual database file)
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    
    # Execute the insert command with string values
    try:
        c.execute('''
            INSERT INTO resumes (name, experience, skills, education, achievements)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            parsed_resume_details['name'],
            parsed_resume_details['experience'],
            skills_str,         # Use the string of skills
            parsed_resume_details['education'],
            achievements_str     # Use the string of achievements
        ))
        conn.commit()  # Commit the transaction
    except Exception as e:
        print(f"An error occurred: {e}")  # Print the error message for debugging
    finally:
        conn.close()  # Ensure the connection is closed

def query_db(query, cursor):
    cursor.execute(query)
    rows = cursor.fetchall()
    return rows

# Define the SQL query function
def sql_wrapper(query):
    print(f"Query here is ##{query}##")
    query = query[query.index("SELECT"):]  # Parse out the generated SQL query
    conn = sqlite3.connect('resumes.db')
    cursor = conn.cursor()
    rows = query_db(query, cursor)  # This function needs to be defined
    conn.close()
    return rows

# Create a query database tool
query_db_tool = Tool(
    name="Query Database",
    func=sql_wrapper,
    description="Useful for querying the SQL database for datasets"
)

tools = [query_db_tool]

# Update the prompt template to include tool_names
sql_prompt = PromptTemplate.from_template("""
You are an assistant with the following tools: {tools}

Your goal is to query data from an SQLite database, specifically from the 'resumes' table, and return relevant information directly to the user.

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

For example, if the user asks "Who among the applicants has a Python background?", directly query the 'resumes' table, extract the names of the applicants, and return the list of names.

Begin!

New input: {input}
{agent_scratchpad}
""")



# Create a comma-separated list of tool names for the prompt
tool_names = ', '.join(tool.name for tool in tools)

# Create the model
model = ChatOpenAI(model="gpt-4o-mini")  # Correct model initialization

# Create the agent
agent = create_react_agent(model, tools, sql_prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit app UI
st.title("AI-Powered Resume Filtering with GPT-4")

# Create the database if it doesn't exist
create_resume_db()

# Upload resumes
uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX/Text)", accept_multiple_files=True)

# Display resume processing
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        
        # Extract text from the uploaded file
        resume_text = extract_text_from_file(uploaded_file)
        
        # Extract structured resume details using GPT-4
        structured_resume_details = extract_resume_details_with_agent(resume_text)
        
        # Parse the extracted text into structured data
        parsed_resume_details = parse_extracted_text(structured_resume_details)
        
        # Insert the resume into the database
# Insert the resume into the database
        if parsed_resume_details:
            insert_resume_into_db(parsed_resume_details)
            st.success(f"Resume data for {parsed_resume_details['name']} inserted into the database.")

            # Confirm data insertion
            st.write("Current resumes in the database:")
            
            # Create a new database connection and cursor
            conn = sqlite3.connect('resumes.db')
            cursor = conn.cursor()
            
            current_resumes = query_db("SELECT * FROM resumes", cursor)  # Fetch all resumes for confirmation
            for resume in current_resumes:
                st.write(resume)  # Display the current resumes

            # Close the database connection
            conn.close()
        else:
            st.error("Failed to parse and insert resume data.")



conn = sqlite3.connect('resumes.db')
cursor = conn.cursor()
current_resumes = query_db("SELECT * FROM resumes", cursor)  # Fetch all resumes for confirmation

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to handle user queries and generate responses
def query_db_tool(query):
    conn = sqlite3.connect('resumes.db')  # Adjust with your database path
    cursor = conn.cursor()
    
    # Use the query to fetch data from the database
    cursor.execute(f"SELECT * FROM resumes WHERE name LIKE '%{query}%'")  # Example query
    results = cursor.fetchall()
    
    conn.close()
    
    return results

# Chat interface
# Chat interface
st.title("Resume Query Chatbot")

# Display chat messages
for message in st.session_state.messages:
    st.markdown(f"**{message['role']}:** {message['content']}")

# User input
user_input = st.text_input("Ask your question:", "")

# On submit
if st.button("Send"):
    if user_input:
        # Log the user message in the session state
        st.session_state.messages.append({"role": "User", "content": user_input})

        # Send the user input to the agent executor
        response = agent_executor.invoke({"input": user_input})

        # Log the assistant's response
        st.session_state.messages.append({"role": "Assistant", "content": response})

        # Display the assistant's response
        st.markdown(f"**Assistant:** {response}")
