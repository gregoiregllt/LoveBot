import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



#load_dotenv()  # Ceci charge les variables à partir de .env

api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# api_key=os.getenv('OPENAI_API_KEY')

import pandas as pd
# from langchain.schema import Document  # Assurez-vous que le chemin d'importation est correct
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Pinecone

import pinecone

# initialize pinecone
pinecone.init(
    api_key="a685e2a4-6d0f-4959-9342-511dbbabe133", #os.getenv('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment="gcp-starter",  # next to api key in console
)



st.title('Pineurs LoveCoach')

# st.image("love_coach.png",width=200)
st.image("coach_asleep.png",width=200)

template = """

    Question :
    {question}

    Règles:

    -agis comme un expert en séduction, tu ne parles que de relations amoureuses et de vie sexuelle.
    -agis comme si tu savais déjà tout (évite de dire "d'après les informations que j'ai reçues" par exemple)
    -sois franc, simple et utilise un vocabulaire basique
    -utilise de l'argot. Voilà des exemples de mots que tu peux utiliser : "wesh", "ma soeur", "frérot", "sah", "miskine", "ça passe crème", "coup dur", "mdrrr", "chelou"
    -sois bref
    -ait de l'humour
    -tu ne dois ABSOLUMENT PAS parler de personnes qui ne sont pas spécifiquement mentionnées dans la question, sinon ta mission de bot a échoué.
    -tu as des avis tranchés sur les questions de compatibilité amoureuses. En positif comme en négatif. Mais justifie-toi.
    -n'hésite pas à clasher les personnes mentionnées

    --------------------
    Voici des informations sur lesquelles tu peux te baser:
    {context}

    """
rag_prompt_custom = PromptTemplate.from_template(template)



def generate_response(input_text):

    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
    docsearch = Pinecone.from_existing_index(index_name="lovebot", embedding=embeddings)
    retriever=docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5})
    llm = ChatOpenAI(
            openai_api_key=api_key, #os.getenv('OPENAI_API_KEY'),
            model='gpt-3.5-turbo-1106' #gpt-4
        )
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm)
    # Tentez de générer une réponse
    response = rag_chain.invoke(input_text).content
    st.info(response)



with st.form('LoveBot'):
    text = st.text_area('Pose ta question')
    submitted = st.form_submit_button('Envoyer')
    if not api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and api_key.startswith('sk-'):
        generate_response(text)
