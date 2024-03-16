import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
#from openai import OpenAI

#Sidebar contents
with st.sidebar:
    st. title('PDF Reader')
    st.markdown('''
                A customised conversational chatbot for your PDF
                ''')
    #add_vertical_space(5)
    st.write('A Generative AI at your service')

#load_dotenv()
def main():
    st.header("Upload your PDF and know about it")

    load_dotenv()

    #upload a PDF file
    pdf=st.file_uploader("Upload your PDF", type='pdf')
    #st.write(pdf.name)

    #st.write(pdf)
    if pdf is not None:
        pdf_filename = pdf.name
        pdf_reader = PdfReader(pdf)
        #st.write(pdf_reader)
        text=""
        for page in pdf_reader.pages:
            text+= page.extract_text()
            #st.write(text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        #st.write(chunks)
        
        store_name=pdf_filename[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            #st.write('Embeddings Loaded from Disk')
        else:
            embeddings = OpenAIEmbeddings()

            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        
        #Accept user question/query
        query = st.text_input("Ask question about your file: ")
        #st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)




            #st.write(docs)

        

if __name__=='__main__':
    main()