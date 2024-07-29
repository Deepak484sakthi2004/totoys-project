from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import os
import whisper
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import HumanMessage, AIMessage
import warnings

from db.pgService import getContext
from voice.audioToText import audio_to_transcript

app = Flask(__name__)
app.secret_key = 'your_secret_key' 
#memory = ConversationBufferMemory(memory_key="chat_history",input_key="user_question")

# Set up the embedding model
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} 

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

def load_audio_model():
    model = whisper.load_model("base")
    return model

def prettify_text(text):
    formatted_text = text.replace('***', '<strong><em>').replace('**', '<strong>').replace('*', '<em>').replace('<strong><em>', '<strong><em>').replace('<em>', '</em><br>').replace('</em>', '<em>')
    formatted_text = formatted_text.replace('</em><br>', '</em><br><br>').replace('<em><strong>', '<br><strong>').replace('<strong>', '</strong>')
    return formatted_text

# def get_conversational_chain(expertLevel):
#     prompt_template = f"""
#         Chat History: {chat_history}
#         system
#         You are an assistant for question-answering tasks.
#         Use the following pieces of retrieved context to answer the question.
#         Use three points minimum and keep the answer concise. <eot_id><start_header_id>user<end_head_id>
#         Question: {user_question}
#         Context: {context}
#         Student_expertise_level: {expertLevel}
#         Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question", "chat_history"])
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyClo-Pfhrww33nYHWXNW_UbjXa7pVggGtM')
#     chain = LLMChain(
#         llm=model,
#         prompt=prompt,
#         verbose=True,
#         memory=memory,
#     )
#     return chain

# def generate_answer_llm(user_question, expertLevel):
#     context = getContext(session['class'], user_question)
#     chain = get_conversational_chain(expertLevel)
#     response = chain.predict(input=user_question, context=context)
#     return response, context



# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_question")

def get_conversational_chain(expertLevel, user_question, context):
    prompt_template = """
        Chat History: {chat_history}
        system
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        Use three points minimum and keep the answer concise.
        <eot_id><start_header_id>user<end_head_id>
        Question: {user_question}
        Context: {context}
        Student_expertise_level: {expertLevel}
        Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question", "chat_history", "expertLevel"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyClo-Pfhrww33nYHWXNW_UbjXa7pVggGtM')
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    return chain

def generate_answer_llm(user_question, expertLevel):
    chat_history = memory.load_memory_variables({"user_question": user_question})['chat_history']
    context = getContext(session['class'], user_question)
    chain = get_conversational_chain(expertLevel, user_question, context)
    response = chain.predict(context=context, user_question=user_question, chat_history=chat_history, expertLevel=expertLevel)
    print("THE RESPONSE OF THE QUESTION IS \n: ")
    print(response)
    print(type(response))
    return response

################



def get_subject_classification_chain():
    prompt_template = """
        You are an assistant for classifying questions into subjects.
        The subjects are: English, Maths, Science, Social Science.

        Below is the question:
        Question: {user_question}

        Identify the subject of the question from the provided list of subjects.
        The subject should be one of the following: English, Maths, Science, Social Science.

        Subject: return only the subject name
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyClo-Pfhrww33nYHWXNW_UbjXa7pVggGtM')
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True,
    )
    return chain

def find_subject(user_question):
    chain = get_subject_classification_chain()
    response = chain.predict(user_question=user_question)
    print("THE DETECTED SUBJECT IS :",response)
    return response.strip()
    



from flask import Flask, request, jsonify, render_template, session, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def main_page():
    if 'username' not in session or 'class' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        user_class = data.get('class')
        if username and user_class:
            session['username'] = username
            session['class'] = user_class
            return jsonify({'message': 'Login successful!'}), 200
        else:
            return jsonify({'message': 'Invalid data!'}), 400
    return render_template('login.html')

@app.route('/audio')
def audio_page():
    english = request.args.get('english')
    maths = request.args.get('maths')
    science = request.args.get('science')
    socialScience = request.args.get('socialScience')
    return render_template('recordAudio1.html', english=english, maths=maths, science=science, socialScience=socialScience)

@app.route('/qa', methods=['GET', 'POST'])
def qa_page():
    if request.method == 'GET':
        english = request.args.get('english')
        maths = request.args.get('maths')
        science = request.args.get('science')
        socialScience = request.args.get('socialScience')
        return render_template('qa.html', english=english, maths=maths, science=science, socialScience=socialScience)
    elif request.method == 'POST':
        data = request.get_json()
        question = data['question']
        english = int(data.get('english', 0))
        maths = int(data.get('maths', 0))
        science = int(data.get('science', 0))
        socialScience = int(data.get('socialScience', 0))

        subject = find_subject(question)  # Determine the subject
        generated_text = generate_answer_llm(question, subject)
        return jsonify({'generated_text': prettify_text(generated_text)})

@app.route('/save_audio', methods=['POST'])
def save_audio():
    audio_data = request.files['audio']
    english = int(request.form['english'])
    maths = int(request.form['maths'])
    science = int(request.form['science'])
    socialScience = int(request.form['socialScience'])

    audio_filename = 'recorded_audio.wav'
    audio_data.save(audio_filename)
    audio_model = load_audio_model()
    transcript = audio_to_transcript(audio_filename, audio_model)
    
    subject = find_subject(transcript)  # Determine the subject
    generated_text = generate_answer_llm(transcript, subject)
    return jsonify({'generated_text': prettify_text(generated_text)})

if __name__ == '__main__':
    app.run(debug=True)
