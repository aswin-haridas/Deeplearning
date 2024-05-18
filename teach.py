import json
import random
from helpers import remove_duplicate_words

def load_knowledge_base(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            data: dict = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"questions": []}

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def save_vocab(file_path: str, question: str, answer: str):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f' {answer}')

def generate_question(file_path: str, num_words: int) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as vocab_file:
            vocab = vocab_file.read().split()
            if len(vocab) < num_words:
                return "Not enough words in vocabulary."
            
            random.shuffle(vocab)
            question = ' '.join(vocab[:num_words])
        return question
    except FileNotFoundError:
        return "Vocabulary file not found."

def chat_bot():
    knowledge_base_file = 'knowledge_base.json'
    vocab_file = 'vocab.txt'
    knowledge_base: dict = load_knowledge_base(knowledge_base_file)

    while True:
        question: str = generate_question(vocab_file, 1)
        print(f'Bot: enthanu {question}')
        user_answer: str = input('Your answer: ')
        if user_answer.lower() == 'quit':
            break
        if user_answer.lower() != '!!':
            knowledge_base["questions"].append({"question": question, "answer": user_answer})
            save_knowledge_base(knowledge_base_file, knowledge_base)
            save_vocab(vocab_file, question, user_answer)
            remove_duplicate_words(vocab_file)

if __name__ == '__main__':
    chat_bot()