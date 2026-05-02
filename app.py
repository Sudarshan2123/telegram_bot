import base64
import traceback
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Request
import httpx
from logging import Logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from flow_graph import create_flow_graph
from state_node import AppState
import psutil
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

state = AppState()
tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()


@asynccontextmanager
async def lifespan(app: FastAPI):
    process = psutil.Process(os.getpid())

    print(f"Memory at start: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    state.llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=1024
    )
    print(f"After LLM init: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    state.chatllm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=1024
    )
    print(f"After ChatLLM: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    state.agent = create_flow_graph(state.chatllm, state.llm)
    print(f"After agent: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    await tg_app.initialize()

    webhook_url = f"{os.getenv('RENDER_URL')}/telegram/{TELEGRAM_BOT_TOKEN}"

    await tg_app.bot.set_webhook(
        url=webhook_url,
        drop_pending_updates=True
    )

    info = await tg_app.bot.get_webhook_info()
    if info.url == webhook_url:
        print(f"Webhook confirmed: {info.url}")
    else:
        print(f"Webhook mismatch! Expected: {webhook_url} Got: {info.url}")

    yield

    await tg_app.shutdown()
    print("Shutdown complete — webhook kept alive")


app = FastAPI(lifespan=lifespan)


# ── TEXT HANDLER ──────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_msg = update.message.text

    print(f"Text from {chat_id}: {user_msg}")

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        response = await state.agent.ainvoke({
            "messages": [HumanMessage(content=user_msg)]
        })
        reply = response["messages"][-1].content
        print(f"Reply: {reply[:100]}")

    except Exception as e:
        print(f"Error in handle_message: {e}")
        traceback.print_exc()
        reply = "Sorry, something went wrong. Please try again!"

    await update.message.reply_text(reply)


# ── PHOTO HANDLER ─────────────────────────────────────────────
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    print(f"Photo from {chat_id}")

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        caption = update.message.caption or "Analyze this vehicle and suggest insurance."
        Logger.info(f"Processing document as image: {file} with MIME type {image_base64[:30]}...")
        Logger.info(f"Caption: {caption}")
        response = await state.agent.ainvoke({
            "messages": [HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {"type": "text", "text": caption}
            ])]
        })
        reply = response["messages"][-1].content
        print(f"Photo reply generated")

    except Exception as e:
        print(f"Error in handle_photo: {e}")
        traceback.print_exc()
        reply = "Sorry, I couldn't analyze the image. Please try again!"

    await update.message.reply_text(reply)


# ── DOCUMENT HANDLER ──────────────────────────────────────────
async def handle_documents(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    doc = update.message.document
    print(f"📎 Document from {chat_id}: {doc.mime_type}")

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        if doc.mime_type and doc.mime_type.startswith("image/"):
            file = await context.bot.get_file(doc.file_id)
            image_bytes = await file.download_as_bytearray()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            caption = update.message.caption or "Analyze this vehicle and suggest insurance."
            Logger.info(f"Processing document as image: {doc.file_name} with MIME type {doc.mime_type}")
            Logger.info(f"Caption: {caption}")
            response = await state.agent.ainvoke({
                "messages": [HumanMessage(content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{doc.mime_type};base64,{image_base64}"
                        }
                    },
                    {"type": "text", "text": caption}
                ])]
            })
            reply = response["messages"][-1].content
        else:
            reply = "I can only process image files. Please send a vehicle photo!"

    except Exception as e:
        print(f"Error in handle_documents: {e}")
        traceback.print_exc()
        reply = "Sorry, I couldn't process the file. Please try again!"

    await update.message.reply_text(reply)


# ── REGISTER HANDLERS ─────────────────────────────────────────
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
tg_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
tg_app.add_handler(MessageHandler(filters.Document.ALL, handle_documents))


# ── WEBHOOK ENDPOINT ──────────────────────────────────────────
@app.post("/telegram/{token:path}")
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_BOT_TOKEN:
        print(f"Invalid token received: {token}")
        return {"status": "unauthorized"}

    try:
        data = await request.json()
        update = Update.de_json(data, tg_app.bot)
        await tg_app.process_update(update)
    except Exception as e:
        print(f"Webhook processing error: {e}")
        traceback.print_exc()

    return {"status": "ok"}


# ── HEALTH CHECK ──────────────────────────────────────────────
@app.head("/")
@app.get("/") 
async def health_check():
    return {
        "status": "running",
        "bot": "vehicle insurance assistant",
        "agent": "ready" if state.agent else "not initialized"
    }


# @app.get("/debug/webhook")
# async def debug_webhook():
#     """Check webhook status from Render's server."""
#     async with httpx.AsyncClient() as client:
#         response = await client.get(
#             f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo"
#         )
#         return response.json()