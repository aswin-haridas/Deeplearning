from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from teach import *

Token: Final = "7060921917:AAHmCkYBMaPr-oKsbBt684Jl-6ITl3QzZ9Q"  # Replace with your actual bot token

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("hi")

def handle_response(text: str) -> str:
    load_knowledge_base("knowledge_base.json")
    text = get_question("vocab.txt", random.randint(2, 3))
    return text

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text
    # print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')
    response = handle_response(text)
    knowledge_base: dict = load_knowledge_base("knowledge_base.json")
    knowledge_base["questions"].append({"question": response, "answer": text})
    save_knowledge_base("knowledge_base.json", knowledge_base)
    save_vocab("vocab.txt", text, response)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print('Starting bot...')
    app = Application.builder().token(Token).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT , handle_message))
    app.add_error_handler(error)
    print('Polling...')
    app.run_polling(poll_interval=1)