import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def enviar_chute_site(driver, palavra):
    # Preenche o campo
    input_box = driver.find_element(By.CSS_SELECTOR, "input.word")
    input_box.clear()
    input_box.send_keys(palavra)
    input_box.send_keys(Keys.ENTER)

    # Aguarda resposta
    time.sleep(1.5)

    # Verifica se apareceu a mensagem de palavra repetida
    try:
        msg_box = driver.find_element(By.CSS_SELECTOR, ".message-text")
        if "already guessed" in msg_box.text:
            print(f"⚠️ Palavra já testada: {palavra}")
            return palavra, None  # Ignorar
    except:
        pass  # nenhuma mensagem = segue normal

    # Lê o chute mais recente
    try:
        resultado = driver.find_element(By.CSS_SELECTOR, ".row-wrapper.current .row")
        texto = resultado.text.strip()
        palavra_retornada, score = texto.split()
        return palavra_retornada.lower(), int(score)
    except Exception as e:
        print("⚠️ Erro ao tentar ler o resultado:", e)
        return palavra, None
    
def get_embedding(word):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        inputs = tokenizer(word)
        outputs = model(**inputs)
        # Usa o embedding do [CLS] token
        embedding = outputs.last_hidden_state[0][0].numpy()
    return embedding