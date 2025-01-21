from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters

from transformers import pipeline
import torch
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                torch_dtype=torch.bfloat16, device_map="auto")

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hallo! I'm your AI Assistant bot. How can I help you?")

async def process(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text.strip()
    messages = [
        {"role": "system", "content": "You are a chatbot who shares facts!"},
        {"role": "user", "content": user_message},
    ]   
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95) 
    response = outputs[0]["generated_text"].split("</s>")[-1].strip() 
    await update.message.reply_text(response)

def main() -> None:
    API_token = "7790702095:AAGjT1ZOpMcTJQPcX4A5yng4_0QvHzLvI1A"
    application = Application.builder().token(API_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process))
    print("LimBot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()