import streamlit as st
from audiorecorder import audiorecorder
from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor
import torch
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory





class Speech_bot:
    def __init__(self):
        self.model_id="distil-whisper/distil-large-v2"
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model=AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id,low_cpu_mem_usage=True).to(self.device)
        self.processor=AutoProcessor.from_pretrained(self.model_id)
        self.speech_to_text_model=pipeline("automatic-speech-recognition",model=self.model,tokenizer=self.processor.tokenizer,feature_extractor=self.processor.feature_extractor,device=self.device)
       

        
    def speech_to_text(self,recording):
        transcription=self.speech_to_text_model(recording)['text']
        # st.write(transcription)
        return transcription


    def load_llm_model(self):
        model=Ollama(model='llama3.1')
        return model

    def record_audio(self):
        audio=audiorecorder()
        if audio:
            audio.export("./recording.mp3")
            user_question=self.speech_to_text('./recording.mp3')
            return user_question

    

        
    def main(self):
        st.title("Voice Chat-Bot")
        model=self.load_llm_model()
        if "messages" not in st.session_state:
            st.session_state.messages=[]
        with st.container():
            for messages in st.session_state.messages:
                with st.chat_message(messages["role"]):
                    st.markdown(messages["content"])
            with st.chat_message("user"):
                    user_prompt=self.record_audio()
                    if user_prompt:
                        st.session_state.messages.append({"role":"user","content":user_prompt})
                        
                        # st.markdown(str(user_prompt))

                        prompt=ChatPromptTemplate.from_messages(
                [
                    ("system","You are an AI chat bot that will answer all the questions asked by the user. Keep it short and concise."),
                    # ("ai","Hi ! My name is JARVIS. How can I help you today ?")
                    ("human","{input}")
                ]
            )
                        chain=prompt | model
                        result=chain.invoke({"input":user_prompt})
                        if 'answer' not in st.session_state:
                            st.session_state['answer']=result
                            st.session_state.messages.append({"role":"assistant","content":result})
            if 'answer' in st.session_state:            
                with st.chat_message("assistant"):
                    st.markdown(str(user_prompt))
                    st.markdown(st.session_state['answer'])
                    del st.session_state['answer']


if __name__=='__main__':
    bot=Speech_bot()
    bot.main()





