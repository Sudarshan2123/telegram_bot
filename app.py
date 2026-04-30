import base64
import os
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# Ensure these files are in the same directory as main.py
from flow_graph import create_flow_graph
from state_node import AppState

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RENDER_URL = os.getenv("RENDER_URL")

state = AppState()

# Define tg_app globally so it's accessible by the webhook and handlers
tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize LLMs
    global _graph
    state.llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=2048
    )
    state.chatllm = ChatGroq(
        model="llama-3.2-90b-text-preview",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2048
    )
    _graph = create_flow_graph(state.chatllm)

    # Register Telegram webhook
    await tg_app.initialize()
    if RENDER_URL:
        webhook_url = f"{RENDER_URL.rstrip('/')}/telegram/{TELEGRAM_BOT_TOKEN}"
        await tg_app.bot.set_webhook(webhook_url)
        print(f"Webhook set to {webhook_url}")
    else:
        print("WARNING: RENDER_URL not found. Webhook not set.")

    yield

    # Cleanup
    await tg_app.bot.delete_webhook()
    await tg_app.shutdown()
    print("Server shutdown complete")

app = FastAPI(lifespan=lifespan)

# ── TEXT HANDLER ─────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    chat_id = update.message.chat_id
    user_msg = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    response = await _graph.ainvoke({
        "messages": [HumanMessage(content=user_msg)]
    })
    agent_response = response["messages"][-1].content
    await update.message.reply_text(agent_response)

# ── PHOTO HANDLER ─────────────────────────────────────────────
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    
    # Using BytesIO to avoid saving files to Render's ephemeral disk
    out = io.BytesIO()
    await file.download_to_memory(out)
    image_bytes = out.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    caption = update.message.caption or "Analyse the photo and provide insights."

    response = await _graph.ainvoke({
        "messages": [HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            },
            {"type": "text", "text": caption}
        ])]
    })
    await update.message.reply_text(response["messages"][-1].content)

# ── DOCUMENT HANDLER ──────────────────────────────────────────
async def handle_documents(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    doc = update.message.document  

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    if doc.mime_type and doc.mime_type.startswith("image/"):
        file = await context.bot.get_file(doc.file_id)
        out = io.BytesIO()
        await file.download_to_memory(out)
        image_bytes = out.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        caption = update.message.caption or "Describe this image"

        response = await _graph.ainvoke({
            "messages": [HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{doc.mime_type};base64,{image_base64}"}
                },
                {"type": "text", "text": caption}
            ])]
        })
        agent_response = response["messages"][-1].content
    else:
        agent_response = "Sorry, I can only process image documents at the moment."

    await update.message.reply_text(agent_response)

# ── REGISTER HANDLERS ─────────────────────────────────────────
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
tg_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
tg_app.add_handler(MessageHandler(filters.Document.ALL, handle_documents))

# ── WEBHOOK ENDPOINT ──────────────────────────────────────────
@app.post(f"/telegram/{TELEGRAM_BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"status": "ok"}