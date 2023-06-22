

with open("RickAndMortyScripts.csv", "r", encoding='utf-8') as file:
                text = file.read()
# from typing import Annotated
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing_extensions import Annotated

from fastapi import FastAPI,Depends,HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import base64
from jose import JWTError, jwt
from passlib.context import CryptContext
security = HTTPBearer()
app = FastAPI()

VALID_USERNAME = 'admin'
VALID_PASSWORD = 'password'
SECRET_KEY = '1234567'
ALGORITHM = 'HS256'

pwd_context = CryptContext(schemes=['bcrypt'])

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username, password):
    if username == VALID_USERNAME and verify_password(password, get_password_hash(VALID_PASSWORD)):
        return True
    return False

def create_access_token(username):
    payload = {'sub': username}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload['sub']
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid token',
            headers={'WWW-Authenticate': 'Bearer'},
        )

def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    username = decode_token(token)
    return username

@app.post('/login')
def login(username: str, password: str):
    if authenticate_user(username, password):
        access_token = create_access_token(username)
        return {'access_token': access_token}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password',
            headers={'WWW-Authenticate': 'Bearer'},
        )
                

@app.get("/botquestion/{question}")
def read_item(question: str,username: str = Depends(authenticate)):
        
                # print(text)
      

# model_name = "microsoft/DialoGPT-large"
        model_name = "microsoft/DialoGPT-medium"
# model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        chat_history_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")

# chatting 5 times with nucleus sampling & tweaking temperature
        for step in range(5):
    # take user input
        #     text = input(">> You:")
    # encode the input and add end of string token
            input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors="pt")
    # concatenate new user input with chat history (if there is)
            bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
            print(chat_history_ids)

    # generate a bot response
            chat_history_ids = model.generate(
                    bot_input_ids,
                    max_length=1000,
                    do_sample=False,
                    top_p=0.70,
                    top_k=100,
                    temperature=0.80,
                    num_return_sequences=1,
                    length_penalty=1,
                    repetition_penalty=1.5,
                    early_stopping=True ,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id
             )
    #print the output
            output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            return(f"{output}")
