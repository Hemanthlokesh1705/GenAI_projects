from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from pathlib import Path
from google import genai
import time
load_dotenv()
class DocumentRetrieval:
    def __init__(self,model_name:str="gemini-2.5-flash-lite",file_path:str="../Data/My_resume.pdf"):
        self.path=Path(file_path)
        self.model=None
        self.model_name=model_name
        self.client=None
        self.my_file=None
        self.initalize_Model()
        self.process_file()
    def initalize_Model(self):
        if not self.path.exists():
            raise ValueError(f"Given file_path {self.path} not exists")
        try:
            self.model=ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.1,
                max_tokens=200
            )
            if self.model is None:
                raise RuntimeError("Error in loading model")
            print("Model setup is successfull")
        except Exception as e:
            print(e)
    def process_file(self):
        self.client=genai.Client()
        if self.client is None:
            raise RuntimeError
        self.my_file=self.client.files.upload(file=self.path)
        while self.my_file.state.name=="PROCESSING":
            time.sleep(2)
            self.my_file=self.my_file.files.get(name=self.my_file.name)
        if self.my_file.state.name != "ACTIVE":
            raise RuntimeError("File processing failed")
        print("file Proccessed Succesfully..")
    def retrive_information(self,user_prompt:str):
        message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "file",
                        "file_id": self.my_file.uri,
                        "mime_type": "application/pdf",
                    },
                ]
            )

        response = self.model.invoke([message])
        return response
def main():
    document_retrival=DocumentRetrieval()
    print(document_retrival.retrive_information("The document belongs to who"))
if __name__=="__main__":
    main()

        

