import os
import streamlit as st
import yt_dlp as youtube_dl
import whisper
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from uuid import uuid4
import glob

# Streamlit layout setup
st.title("📄 YouTube Video and Stanford CS229 Machine Learning Question Answering")
st.write("Upload a YouTube video or query Stanford CS229 Machine Learning. Provide your OpenAI API key and Pinecone API key to start.")

# OpenAI API Key input
openai_api_key = st.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.text_input("Pinecone API Key", type="password")

if not openai_api_key or not pinecone_api_key:
    st.info("Please add your OpenAI and Pinecone API keys to continue.", icon="🗝️")
else:
    # Initialize OpenAI API key
    import openai
    openai.api_key = openai_api_key

    # Initialize Pinecone API key
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Choose the task
    task = st.radio("Choose an option:", ("Query Stanford", "Upload YouTube Video"))

    if task == "Query Stanford":
        # Question input
        question = st.text_area("Ask a question about the transcripts!", placeholder="Can you give me a short summary?")
        if question:
            # Connect to existing Pinecone index
            index_name = 'transcription-index'
            index = pc.Index(index_name)
            
            # Querying
            embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
            vectorstore = LangchainPinecone(index, embed.embed_query, "transcript")
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.0)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
            
            answer = qa.invoke(question)
            st.write("Answer:", answer)

    elif task == "Upload YouTube Video":
        # Input YouTube URL
        youtube_url = st.text_input("Enter YouTube URL")

        if youtube_url:
            st.write("Downloading and transcribing video...")

            # Directory setup
            audio_output_dir = "files/audio"
            transcript_output_dir = "files/transcripts"
            os.makedirs(audio_output_dir, exist_ok=True)
            os.makedirs(transcript_output_dir, exist_ok=True)

            # Download video and extract audio
            ydl_config = {
                "format": "bestaudio/best",
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
                "outtmpl": os.path.join(audio_output_dir, "%(title)s.%(ext)s"),
                "continuedl": False,
            }

            audio_filename = ""
            try:
                with youtube_dl.YoutubeDL(ydl_config) as ydl:
                    info_dict = ydl.extract_info(youtube_url, download=True)
                    audio_filename = ydl.prepare_filename(info_dict).replace(".m4a", ".mp3")
            except youtube_dl.DownloadError as e:
                st.error(f"Error downloading {youtube_url}: {str(e)}")

            if audio_filename:
                # Transcribe audio file
                model = whisper.load_model("base")
                transcript = ""
                
                try:
                    output_file = os.path.join(transcript_output_dir, f"{os.path.basename(audio_filename)}.txt")
                    audio = whisper.load_audio(audio_filename)
                    result = model.transcribe(audio)
                    transcript = result["text"]
                    with open(output_file, "w") as f:
                        f.write(transcript)
                    st.write(f"Transcript saved to {output_file}")
                except Exception as e:
                    st.error(f"Error processing {audio_filename}: {e}")

                if transcript:
                    # Question input
                    question = st.text_area("Ask a question about the transcribed video!", placeholder="Can you give me a short summary?")
                    if question:
                        # Create a new Pinecone index for the YouTube video transcript
                        youtube_index_name = 'youtube'
                        if youtube_index_name not in [index["name"] for index in pc.list_indexes()]:
                            spec = ServerlessSpec(cloud="aws", region="us-east-1")
                            pc.create_index(youtube_index_name, dimension=1536, metric='dotproduct', spec=spec)
                        
                        youtube_index = pc.Index(youtube_index_name)

                        # Prepare text for indexing
                        tokenizer = tiktoken.get_encoding('cl100k_base')
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20, length_function=lambda text: len(tokenizer.encode(text)))
                        embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)

                        # Indexing
                        texts, metadatas = [], []
                        chunks = text_splitter.split_text(transcript)
                        for chunk_id, chunk in enumerate(chunks):
                            texts.append(chunk)
                            metadatas.append({"chunk": chunk_id, "file": os.path.basename(audio_filename)})

                        ids = [str(uuid4()) for _ in range(len(texts))]
                        embeds = embed.embed_documents(texts)
                        youtube_index.upsert(vectors=zip(ids, embeds, metadatas))

                        # Querying
                        vectorstore = LangchainPinecone(youtube_index, embed.embed_query, "transcript")
                        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.0)
                        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
                        
                        answer = qa.invoke(question)
                        st.write("Answer:", answer)

                        # Delete the Pinecone index after use
                        pc.delete_index(youtube_index_name)
                        st.write("Temporary Pinecone index 'youtube' deleted.")
