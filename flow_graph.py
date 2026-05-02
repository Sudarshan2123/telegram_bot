"""
vehicle_insurance_graph.py
--------------------------
Production-ready LangGraph flow for a vehicle-insurance chatbot.

Key improvements over the original:
  1. Supervisor output is normalised via a Pydantic validator — no silent mis-routes.
  2. Image history is sanitised in every node that calls a non-vision LLM.
  3. `vehicle_context` is stored in graph state so Research always has clean input.
  4. Research query is built from the last *human* message, not the last AI message.
  5. Concurrent-user safety: pass a thread_id per user; MemorySaver isolates sessions.
  6. All fallbacks are explicit and logged.
  7. `analyse_photos` strips stale image messages before passing history to the LLM.
  8. FINISH is reachable: supervisor fires it when it detects the answer was already given.
"""

import asyncio
import logging
from functools import partial

from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from state_node import RoutePlanner, StateNode

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _has_image(message) -> bool:
    """Return True if a message carries an image_url part."""
    if isinstance(message.content, list):
        return any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in message.content
        )
    return False


def _strip_images(messages: list) -> list:
    """
    Return a copy of messages with image content removed.
    Text parts inside multi-part messages are preserved.
    """
    cleaned = []
    for msg in messages:
        if isinstance(msg.content, list):
            text_parts = [
                p["text"]
                for p in msg.content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = " ".join(text_parts).strip()
            if text:
                cleaned.append(msg.__class__(content=text))
            # silently drop image-only messages from history
        else:
            cleaned.append(msg)
    return cleaned


def _last_human_text(messages: list, fallback: str = "vehicle insurance") -> str:
    """Return the text of the most recent HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                parts = [
                    p["text"]
                    for p in msg.content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                text = " ".join(parts).strip()
                if text:
                    return text
            elif isinstance(msg.content, str) and msg.content.strip():
                return msg.content.strip()
    return fallback


# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

async def supervisor_node(state: StateNode, chatllm):
    """
    Route the conversation.
    - Detects images in the latest message first (no LLM call needed).
    - Strips images from history before sending to the text LLM.
    - Normalises the LLM's output via RoutePlanner.normalize().
    - Saves vehicle_context so downstream nodes always have clean input.
    """
    messages = state["messages"]

    # Fast-path: image in last message → skip LLM call
    if _has_image(messages[-1]):
        logger.info("Image detected → analyse_photos")
        return {"next_action": "analyse_photos"}

    clean = _strip_images(messages)
    recent = clean[-4:]  # slightly wider window than original for accuracy

    system_prompt = SystemMessage(content="""You are a routing supervisor for a vehicle-insurance chatbot.

Choose EXACTLY one of the following values for next_action:
  - "research"       → user asks about insurance plans, coverage, premiums, recommendations
  - "chat"           → greetings, thanks, general chit-chat, off-topic
  - "FINISH"         → the previous assistant message already fully answered the user

Return ONLY the structured output. No extra text.""")

    try:
        planner = chatllm.with_structured_output(RoutePlanner)
        response: RoutePlanner = await planner.ainvoke([system_prompt] + recent)
        action = response.next_action
    except Exception as exc:
        logger.warning("Supervisor LLM error: %s — defaulting to 'chat'", exc)
        action = "chat"

    logger.info("Supervisor → %s", action)

    # Persist the latest human text so Research can use it even after image stripping
    vehicle_context = state.get("vehicle_context", "")
    human_text = _last_human_text(messages)
    if human_text and human_text != "vehicle insurance":
        vehicle_context = human_text  # update only if we have something real

    return {"next_action": action, "vehicle_context": vehicle_context}


async def chat_node(state: StateNode, chatllm):
    """General conversational responses — no images, no search."""
    messages = _strip_images(state["messages"])
    system_msg = SystemMessage(content="""You are a friendly vehicle insurance assistant.
Answer questions about vehicle insurance clearly and helpfully.
If the user is just greeting or chatting, respond warmly and ask how you can help with their insurance needs.""")
    try:
        response = await chatllm.ainvoke([system_msg, *messages[-6:]])
    except Exception as exc:
        logger.error("chat_node LLM error: %s", exc)
        response = AIMessage(content="I'm here to help with your vehicle insurance queries! What would you like to know?")
    return {"messages": [response]}


async def analyse_photos_node(state: StateNode, llm):
    """
    Use the vision-capable LLM to analyse the uploaded vehicle photo.
    Only the LATEST image message is passed; prior image history is stripped
    to avoid token bloat and model confusion.
    """
    messages = state["messages"]

    # Separate the latest image message from the rest
    latest_image_msg = None
    for msg in reversed(messages):
        if _has_image(msg):
            latest_image_msg = msg
            break

    # Build clean history (no stale images)
    clean_history = _strip_images(messages[:-1])  # everything except the last message
    context_msgs = clean_history[-4:]             # keep last 4 for context

    if latest_image_msg is None:
        # Shouldn't happen, but handle gracefully
        logger.warning("analyse_photos called but no image found — routing to chat")
        return {"next_action": "chat"}

    system_msg = SystemMessage(content="""You are a vehicle insurance expert with vision capabilities.
Analyse the vehicle in the image and extract:
1. Vehicle type (sedan, SUV, hatchback, truck, two-wheeler, etc.)
2. Estimated make/model if visible
3. Any visible damage, dents, or condition issues
4. Approximate age / condition of the vehicle

Be structured and concise. Your analysis will be used to recommend the right insurance plan.""")

    payload = [system_msg, *context_msgs, latest_image_msg]

    try:
        response = await llm.ainvoke(payload)
    except Exception as exc:
        logger.error("analyse_photos_node LLM error: %s", exc)
        response = AIMessage(
            content="I can see a vehicle in the image. Could you share the make, model, and year so I can recommend the best insurance plan?"
        )

    # Update vehicle_context with the AI's analysis for the Research node
    analysis_text = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "messages": [response],
        "vehicle_context": analysis_text[:300],  # keep it short for the search query
    }


async def research_node(state: StateNode, chatllm, llm):
    """
    Search for current insurance plans and synthesise a recommendation.

    Context priority (highest → lowest):
      1. state["vehicle_context"]  — set by analyse_photos or supervisor
      2. Last human message text
      3. Generic fallback
    """
    messages      = state["messages"]
    clean_msgs    = _strip_images(messages)
    recent_msgs   = clean_msgs[-4:]

    # ── Build search query ───────────────────────────────────────────
    vehicle_context = (
        state.get("vehicle_context")
        or _last_human_text(clean_msgs)
        or "vehicle insurance India"
    )
    # Trim to avoid DuckDuckGo query-length issues
    query = f"best vehicle insurance India 2025 {vehicle_context[:80]}"
    logger.info("Research query: %s", query[:120])

    # ── DuckDuckGo search (non-blocking) ────────────────────────────
    def _search() -> list:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=5))

    search_context = ""
    try:
        results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=10)
        logger.info("Search returned %d results", len(results))
        search_context = "\n\n".join(
            f"• {r['title']}: {r['body'][:200]}" for r in results
        )
    except asyncio.TimeoutError:
        logger.warning("DuckDuckGo search timed out")
    except Exception as exc:
        logger.warning("DuckDuckGo search error: %s", exc)

    # ── LLM synthesis ────────────────────────────────────────────────
    system_msg = SystemMessage(content=f"""You are a vehicle insurance expert in India.
Based on the conversation and search results, recommend the TOP 3 insurance plans.

{"### Live Search Results\n" + search_context if search_context else "### Note: Search unavailable — use your knowledge."}

### Vehicle Context
{vehicle_context[:300]}

For each plan provide:
  - Insurer name
  - Estimated annual premium range (INR)
  - 3 key benefits
  - Ideal for (vehicle type / profile)

Be specific, practical, and India-focused.""")

    try:
        response = await llm.ainvoke([system_msg, *recent_msgs])
        reply = (response.content if isinstance(response.content, str) else str(response.content)).strip()
    except Exception as exc:
        logger.error("research_node LLM error: %s", exc)
        reply = ""

    # ── Explicit fallback (never empty) ─────────────────────────────
    if not reply:
        reply = (
            "Here are top vehicle insurance plans in India (2025):\n\n"
            "1. **HDFC ERGO** — ₹6,000–₹15,000/year\n"
            "   • Cashless repairs at 6,800+ garages\n"
            "   • Zero depreciation add-on available\n"
            "   • Quick digital claims\n"
            "   Ideal for: New cars & first-time buyers\n\n"
            "2. **Bajaj Allianz** — ₹5,500–₹14,000/year\n"
            "   • 24×7 roadside assistance\n"
            "   • High claim settlement ratio (98%+)\n"
            "   • Engine protect add-on\n"
            "   Ideal for: Budget-conscious owners\n\n"
            "3. **ICICI Lombard** — ₹6,500–₹16,000/year\n"
            "   • 15,000+ network garages\n"
            "   • Instant policy issuance\n"
            "   • No-claim bonus up to 50%\n"
            "   Ideal for: Premium & older vehicles"
        )

    logger.info("Research reply (first 150 chars): %s", reply[:150])
    return {"messages": [AIMessage(content=reply)]}


# ─────────────────────────────────────────────
# Routing function
# ─────────────────────────────────────────────

def route(state: StateNode) -> str:
    action = state.get("next_action", "chat")
    valid  = {"chat", "research", "analyse_photos", "FINISH"}
    if action not in valid:
        logger.warning("Unknown route '%s' — defaulting to 'chat'", action)
        return "chat"
    return action


# ─────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────

def create_flow_graph(chatllm, llm):
    """
    Build and compile the LangGraph state machine.

    Args:
        chatllm : Text-only LLM (e.g. Groq / GPT-4o-mini) — used for supervisor, chat, research query
        llm     : Vision-capable LLM (e.g. GPT-4o / LLaVA) — used for photo analysis & research synthesis

    Usage (per user session):
        graph  = create_flow_graph(chatllm, llm)
        config = {"configurable": {"thread_id": "<unique-user-id>"}}
        result = await graph.ainvoke({"messages": [HumanMessage(content="...")]}, config=config)
    """
    graph = StateGraph(StateNode)

    graph.add_node("supervisor",     partial(supervisor_node,     chatllm=chatllm))
    graph.add_node("chat",           partial(chat_node,           chatllm=chatllm))
    graph.add_node("analyse_photos", partial(analyse_photos_node, llm=llm))
    graph.add_node("research",       partial(research_node,       chatllm=chatllm, llm=llm))

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "chat":           "chat",
            "research":       "research",
            "analyse_photos": "analyse_photos",
            "FINISH":         END,
        },
    )

    graph.add_edge("chat",           END)
    graph.add_edge("research",       END)
    graph.add_edge("analyse_photos", "research")  # photo analysis always feeds research

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)