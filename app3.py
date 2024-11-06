import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, downlaod_hugging_face_embeddings
from src.prompt import *
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pinecone

# Initialize Streamlit page settings
st.set_page_config(layout="wide")
load_dotenv()

# API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings and Pinecone index
@st.cache_resource
def get_embeddings():
    return downlaod_hugging_face_embeddings()

@st.cache_resource
def load_docsearch():
    embeddings = get_embeddings()
    index_name = "fachatbot"
    return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# Initialize the retrieval and RAG chain
def load_rag():
    docsearch = load_docsearch()
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=500)
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            ('human', '{input}')
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Load RAG model
rag_chain = load_rag()

# Streamlit user interface
#st.title("Ataxia Chatbot")
logo_path = "FARA_LOGO.png"  # Ensure this image file is in your project directory

header_col1, header_col2 = st.columns([4, 5])
with header_col1:
    st.title("Ataxia Chatbot")
      # Adjust width as needed
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.text_input("Age")
    with col2:
        aoo = st.text_input("AOO")
    with col3:
        aims = st.text_input("AIMS")
    with col4:
        mfars = st.text_input("mFARS")

with header_col2:
    st.image(logo_path, width=250)

#input_question = st.text_input("Ask your question")

# Check if any additional information fields have been provided
additional_info = []
if age:
    additional_info.append(f"age = {age}")
if aoo:
    additional_info.append(f"aoo = {aoo}")
if aims:
    additional_info.append(f"aims = {aims}")
if mfars:
    additional_info.append(f"mfars = {mfars}")

# Append additional information to the question if any field is filled
# if additional_info and input_question.strip() != "":
#     input_question += " , given by " + " , ".join(additional_info)

additional_info_text = " , given by " + " , ".join(additional_info) if additional_info else ""

# Main question input field
raw_question = st.text_input("Ask your question")

# Combine the question with additional info text
input_question = raw_question + additional_info_text

# Display the combined question
if additional_info:
    st.write("### Final Question")
    st.write(input_question)

#input_question = st.text_input("Ask your question")
if input_question.strip() != "":
    with st.spinner("Generating Answer..."):
        prediction = rag_chain.invoke({"input": input_question})
    
    answer = prediction["answer"]
    source_documents = prediction["context"]

    # Create two columns for the answer and source documents
    left_col, right_col = st.columns(2)

    # Display answer on the left
    with left_col:
        st.write("### Answer")
        st.write(answer)

    # Display source documents on the right
    with right_col:
        st.write("### Source Documents")
        for document in source_documents:
            content = document.page_content
            page = document.metadata.get('page', 'Unknown Page')
            source = os.path.basename(document.metadata.get('source', 'Unknown Source'))

            with st.container(border=True):
                st.markdown(f"**Content:** {content}")
                st.markdown(f"**Page:** {page}")
                st.write(f"**Paper:** {source}")


#-------------------------------------------------------------------------------------------------


# import os
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_openai import OpenAI
# from langchain_pinecone import PineconeVectorStore
# from src.helper import load_pdf_file, text_split, downlaod_hugging_face_embeddings
# from src.prompt import *
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# import pinecone

# # Initialize Streamlit page settings
# st.set_page_config(layout="wide")
# load_dotenv()

# # API keys
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize embeddings and Pinecone index
# @st.cache_resource
# def get_embeddings():
#     return downlaod_hugging_face_embeddings()

# @st.cache_resource
# def load_docsearch():
#     embeddings = get_embeddings()
#     index_name = "fachatbot"
#     return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# # Initialize the retrieval and RAG chain
# def load_rag():
#     docsearch = load_docsearch()
#     retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#     llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=500)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', system_prompt),
#             ('human', '{input}')
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     return create_retrieval_chain(retriever, question_answer_chain)

# # Load RAG model
# rag_chain = load_rag()

# # Streamlit user interface with logo and title
# logo_path = "FARA_LOGO.png"  # Ensure this image file is in your project directory

# header_col1, header_col2 = st.columns([4, 5])
# with header_col1:
#     st.title("Ataxia Chatbot")

# with header_col2:
#     st.image(logo_path, width=250)

# input_question = st.text_input("Ask your question")
# if input_question.strip() != "":
#     with st.spinner("Generating Answer..."):
#         prediction = rag_chain.invoke({"input": input_question})
    
#     answer = prediction["answer"]
#     source_documents = prediction["context"]

#     # Create two columns for the answer and source documents
#     left_col, right_col = st.columns(2)

#     # Display answer on the left
#     with left_col:
#         st.write("### Answer")
#         st.write(answer)

#     # Display source documents on the right with a bordered container for each document
#     with right_col:
#         st.write("### Source Documents")
#         for idx, document in enumerate(source_documents):
#             content = document.page_content
#             page = document.metadata.get('page', 'Unknown Page')
#             source = os.path.basename(document.metadata.get('source', 'Unknown Source'))
#             pdf_path = os.path.join("D://OpenAI_Projects//FA_Chat_BotData//Data//", source)  # Assuming all PDFs are stored in a folder named 'data'
#             print(pdf_path)

#             # Unique IDs for toggle control
#             pdf_div_id = f"pdf_{idx}"
#             meta_div_id = f"meta_{idx}"

#             # HTML for displaying metadata with a clickable link to toggle PDF view
#             st.markdown(
#                 f"""
#                 <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
#                     <div id="{meta_div_id}">
#                         <p><strong>Content:</strong> {content}</p>
#                         <p><strong>Page:</strong> {page}</p>
#                         <p><strong>Paper:</strong> <a href="javascript:void(0);" onclick="document.getElementById('{meta_div_id}').style.display='none';document.getElementById('{pdf_div_id}').style.display='block';">Open PDF</a></p>
#                     </div>
#                     <div id="{pdf_div_id}" style="display:none;">
#                         <iframe src="{pdf_path}" width="100%" height="400px"></iframe>
#                         <p><a href="javascript:void(0);" onclick="document.getElementById('{pdf_div_id}').style.display='none';document.getElementById('{meta_div_id}').style.display='block';">Close PDF</a></p>
#                     </div>
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )

#---------------------------------------------------------------------------------------------------------
# import os
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_openai import OpenAI
# from langchain_pinecone import PineconeVectorStore
# from src.helper import load_pdf_file, text_split, downlaod_hugging_face_embeddings
# from src.prompt import *
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# import pinecone

# # Initialize Streamlit page settings
# st.set_page_config(layout="wide")
# load_dotenv()

# # API keys
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize embeddings and Pinecone index
# @st.cache_resource
# def get_embeddings():
#     return downlaod_hugging_face_embeddings()

# @st.cache_resource
# def load_docsearch():
#     embeddings = get_embeddings()
#     index_name = "fachatbot"
#     return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# # Initialize the retrieval and RAG chain
# def load_rag():
#     docsearch = load_docsearch()
#     retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#     llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=500)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', system_prompt),
#             ('human', '{input}')
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     return create_retrieval_chain(retriever, question_answer_chain)

# # Load RAG model
# rag_chain = load_rag()

# # Streamlit user interface
# # Display logo and title
# logo_path = "FARA_LOGO.png"  # Ensure this image file is in your project directory

# # Layout for the left section with title, input fields, and answer
# header_col1, header_col2 = st.columns([4, 5])

# # Left column with the Ataxia Chatbot title and patient input fields
# with header_col1:
#     st.title("Ataxia Chatbot")
    
#     # Display input fields for additional patient details
#     st.write("### Enter Patient Details")
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         age = st.text_input("Age")
#     with col2:
#         aoo = st.text_input("AOO")
#     with col3:
#         aims = st.text_input("AIMS")
#     with col4:
#         mfars = st.text_input("mFARS")

#     # Main question input field aligned with "Ataxia Chatbot" column
#     raw_question = st.text_input("Ask your question")

# # Right column for the logo and source documents section
# with header_col2:
#     st.image(logo_path, width=250)
    
#     # Place "Source Documents" title right below the logo
#     st.write("### Source Documents")

# # Collect and format additional information if any fields are provided
# additional_info = []
# if age:
#     additional_info.append(f"age = {age}")
# if aoo:
#     additional_info.append(f"aoo = {aoo}")
# if aims:
#     additional_info.append(f"aims = {aims}")
# if mfars:
#     additional_info.append(f"mfars = {mfars}")

# # Append additional information to the question only if at least one field is filled
# additional_info_text = " , given by " + " , ".join(additional_info) if additional_info else ""

# # Combine the question with additional info text only if any field is provided
# input_question = raw_question + additional_info_text if additional_info else raw_question

# # Display the "Final Question" only if there's additional info to append
# if additional_info:
#     st.write("### Final Question")
#     st.write(input_question)

# # Display Answer and Source Documents sections side-by-side
# answer_col, source_col = st.columns([4, 5])

# # Process the question only if it's not empty
# if input_question.strip():
#     with st.spinner("Generating Answer..."):
#         prediction = rag_chain.invoke({"input": input_question})
    
#     answer = prediction["answer"]
#     source_documents = prediction["context"]

#     # Display answer in the left column
#     with header_col1:
#         st.write("### Answer")
#         st.write(answer)

#     # Display source documents under the logo in the right column
#     with header_col2:
#         for document in source_documents:
#             content = document.page_content
#             page = document.metadata.get('page', 'Unknown Page')
#             source = os.path.basename(document.metadata.get('source', 'Unknown Source'))

#             # Display each document in a bordered container
#             st.markdown(
#                 f"""
#                 <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
#                     <p><strong>Content:</strong> {content}</p>
#                     <p><strong>Page:</strong> {page}</p>
#                     <p><strong>Paper:</strong> {source}</p>
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )



