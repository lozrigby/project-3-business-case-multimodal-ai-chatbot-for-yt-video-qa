![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project III | Business Case: Building a Multimodal AI ChatBot for YouTube Video QA

By Loz Rigby 

Building a chatbot that can translate YouTube videos into text and allow for natural language querying offers several compelling business cases.


### Project Overview

The goal of this final project is to develop an AI bot that combines the power of text and audio processing to answer questions about YouTube videos. The bot will utilize natural language processing (NLP) techniques and speech recognition to analyse both textual and audio input, extract relevant information from YouTube videos, and provide accurate answers to user queries.
The objective of this project is for Stanford students who are in the Machine Learning Class to use the bot to ask questions on all their lectures, to avoid having to search through long lectures and notes. 


### Key Objectives

1.	Develop a text-based question answering (QA) model using pre-trained language models. You may find it useful to fine-tune your model.
2.	Integrate speech recognition capabilities to convert audio into text transcripts.
3.	Build a conversational interface for users to interact with the bot via text.
4.	Retrieve, analyse, and store into a vector database using Pinecone YouTube video content to generate answers to user questions.
5.	Test and evaluate the bot's performance in accurately answering questions about YouTube videos.


### Summary

This project aimed to develop an AI chatbot to help Stanford students efficiently retrieve information from their machine learning lectures on YouTube. By integrating advanced NLP and speech recognition technologies, including Whisper for transcription and GPT-3.5-turbo for language understanding, the system successfully transcribes and processes lecture videos, storing the data in Pinecone for fast, accurate querying. Despite challenges such as transcription accuracy and scalability, the bot was positively evaluated for its ability to answer complex questions and reduce the time spent searching through lecture content, highlighting the potential of AI-driven educational tools.



### Deliverables

1. Source code for the multimodal bot implementation.
2. Detailed report of the project.
3. Presentation slides summarizing the project objectives, process, and results.
4. App folder for deployment (worked well in a virtual environment) to be run on Streamlit.


## Conclusion

This project demonstrates the potential of integrating advanced natural language processing and speech recognition technologies to enhance educational experiences. The objective was to create an AI bot capable of answering questions about Stanford's machine learning lectures on YouTube. By developing this chatbot, the system provides a valuable tool for students to quickly and efficiently retrieve relevant information from their lectures. The project successfully utilized state-of-the-art models like Whisper for transcription and GPT-3.5-turbo for natural language understanding, combined with the scalable vector search capabilities of Pinecone.

The system offers significant benefits, such as reducing the time spent searching through lecture content and providing precise answers to complex queries. It was positively evaluated, demonstrating its effectiveness in addressing the needs of students. However, addressing issues related to transcription accuracy, handling long queries, scalability, and data privacy will be essential for further improvements. Additionally, extending the system's capabilities to handle a broader range of content and queries will enhance its versatility and usefulness.

Overall, this project underscores the importance of leveraging cutting-edge AI technologies in educational settings and sets the stage for future developments in AI-driven educational tools.
