# Advanced Programming Final Project | IT-2308
# Advanced LLM Application
### Team members

1) #### Ainur Ali
2) #### Abdukadyrova Darina
3) #### Akerke Kassymebkova

Telegram Bot Username: t.me/binal_advbot

# Overview

### This project implements a chatbot that can process user queries in two modes:

Web Search Mode: The bot performs a web search using SerpAPI, processes the results, and uses an LLaVA model for analysis.
Document Search Mode: The bot searches pre-uploaded documents stored locally and provides results using a language model.
Additionally, the chatbot analyzes images, generates conclusions, and sends notifications via a Telegram bot.

## Installation
Install special AI models in command prompt
```
ollama run llama3.1
ollama run llama3.2:latest
ollama run llava

```

Execute this command to install necessary packages
```
pip install streamlit requests Pillow beautifulsoup4 langchain langchain-community chromadb serpapi telegram ollama
```

Then, open project folder and write next command in terminal:
```
python -m streamlit run app.py
```
OR
```
streamlit run app.py
```


## Instruction

1) Choose AI model that analyze text (Llama 3.2,etc.)

2) Write search query like to browser. Try not to write complex search queries because search system could misunderstand your query

3) Enter a Query: Type a query in the main text box and press the button to submit it. The bot will fetch the response, analyze images, and generate a conclusion.

4) Telegram Notifications: Responses and notifications will be sent to your configured Telegram chat.

5) Choose type of searching

Text - retrieve only text from internet and analyze it Text model

Image - retrieve only images from internet. Llava model will analyze them.

Both - return text analyze and combined result (text and image)

P.S.: Please, don't write inappropriate words in all queries.




