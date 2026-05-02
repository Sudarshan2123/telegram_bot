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

    # Store the FULL analysis — research_node needs all details to give
    # personalised recommendations. Do NOT truncate here; only the search
    # query (built in research_node) is trimmed to 80 chars.
    analysis_text = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("Photo analysis (%d chars): %s...", len(analysis_text), analysis_text[:120])
    return {
        "messages": [response],
        "vehicle_context": analysis_text,   # full text passed to research_node
    }


async def research_node(state: StateNode, chatllm, llm):
    """
    Search for current insurance plans and synthesise a recommendation
    that is FULLY PERSONALISED to the vehicle analysis from analyse_photos.

    vehicle_context priority (highest → lowest):
      1. state["vehicle_context"]  — set by analyse_photos (full analysis text)
      2. Last human message text
      3. Generic fallback
    """
    messages    = state["messages"]
    clean_msgs  = _strip_images(messages)

    # ── Pull vehicle_context (photo analysis or user text) ───────────
    vehicle_context = (
        state.get("vehicle_context", "").strip()
        or _last_human_text(clean_msgs)
        or "vehicle insurance India"
    )
    logger.info("vehicle_context for research (%d chars): %s...",
                len(vehicle_context), vehicle_context[:120])

    # ── Detect vehicle type from context for smarter query + fallback ─
    ctx_lower = vehicle_context.lower()
    if any(w in ctx_lower for w in ["two-wheeler", "two wheeler", "bike", "motorcycle",
                                     "scooter", "motorbike"]):
        vehicle_type = "two-wheeler"
    elif any(w in ctx_lower for w in ["suv", "fortuner", "creta", "xuv", "innova",
                                       "safari", "harrier"]):
        vehicle_type = "SUV"
    elif any(w in ctx_lower for w in ["truck", "commercial", "lorry", "tempo"]):
        vehicle_type = "commercial vehicle"
    elif any(w in ctx_lower for w in ["sedan", "city", "verna", "dzire", "ciaz"]):
        vehicle_type = "sedan"
    elif any(w in ctx_lower for w in ["hatchback", "swift", "wagnor", "alto",
                                       "i20", "baleno", "polo"]):
        vehicle_type = "hatchback"
    else:
        vehicle_type = "car"

    # Detect damage for add-on recommendations
    has_damage = any(w in ctx_lower for w in ["damage", "dent", "scratch",
                                               "crack", "rust", "worn"])
    is_old     = any(w in ctx_lower for w in ["old", "aged", "high mileage",
                                               "wear", "10 year", "15 year"])

    logger.info("Detected vehicle_type=%s  has_damage=%s  is_old=%s",
                vehicle_type, has_damage, is_old)

    # ── Build targeted search query ──────────────────────────────────
    query = f"best {vehicle_type} insurance India 2025 {vehicle_context[:60]}"
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

    # ── Build add-on advice based on detected condition ───────────────
    addon_advice = ""
    if has_damage:
        addon_advice += "\n- The vehicle has visible damage — recommend Return to Invoice or Repair cover add-ons."
    if is_old:
        addon_advice += "\n- Vehicle appears older — recommend Engine Protection and Consumables add-ons."
    if vehicle_type == "two-wheeler":
        addon_advice += "\n- Two-wheelers need Personal Accident cover for rider + pillion."

    # ── LLM synthesis — full vehicle analysis passed in ──────────────
    system_msg = SystemMessage(content=f"""You are a vehicle insurance expert in India.
The user uploaded a photo of their vehicle. Here is the FULL analysis of that vehicle:

### Vehicle Analysis (from photo)
{vehicle_context}

{"### Live Search Results\n" + search_context if search_context else "### Note: Search unavailable — use your expert knowledge."}

### Your Task
Recommend the TOP 3 insurance plans specifically suited for THIS vehicle.
{addon_advice}

For each plan provide:
  - Insurer name
  - Estimated annual premium range (INR) for this specific vehicle type
  - 3 key benefits relevant to this vehicle
  - Why it suits THIS vehicle specifically (reference the analysis above)

Do NOT give generic recommendations. Tailor everything to the vehicle analysis.
Be specific, practical, and India-focused.""")

    # Include the full clean message history so the LLM has conversation context
    recent_msgs = clean_msgs[-4:]

    try:
        response = await llm.ainvoke([system_msg, *recent_msgs])
        reply = (response.content if isinstance(response.content, str) else str(response.content)).strip()
    except Exception as exc:
        logger.error("research_node LLM error: %s", exc)
        reply = ""

    # ── Smart fallback — vehicle-type aware, never generic ───────────
    if not reply:
        if vehicle_type == "two-wheeler":
            reply = (
                f"Based on your {vehicle_type}, here are the best insurance plans in India (2025):\n\n"
                "1. **Bajaj Allianz Two-Wheeler** — ₹1,500–₹4,000/year\n"
                "   • Personal Accident cover ₹15 lakh\n"
                "   • Pillion rider cover available\n"
                "   • Cashless repairs at 4,000+ garages\n"
                "   Best for: Bikes & scooters up to 150cc\n\n"
                "2. **HDFC ERGO Two-Wheeler** — ₹1,800–₹5,000/year\n"
                "   • Zero depreciation add-on\n"
                "   • 24×7 roadside assistance\n"
                "   • Quick digital claim settlement\n"
                "   Best for: Premium bikes & high-value scooters\n\n"
                "3. **New India Assurance** — ₹1,200–₹3,500/year\n"
                "   • Government-backed reliability\n"
                "   • Wide network across India\n"
                "   • Affordable OD + TP combo\n"
                "   Best for: Budget two-wheelers"
            )
        elif vehicle_type == "SUV":
            reply = (
                f"Based on your {vehicle_type}, here are the best insurance plans in India (2025):\n\n"
                "1. **HDFC ERGO Comprehensive** — ₹12,000–₹28,000/year\n"
                "   • Zero depreciation for higher IDV protection\n"
                "   • Engine protect — critical for SUV off-road use\n"
                "   • 6,800+ cashless garages\n"
                "   Best for: New & mid-aged SUVs\n\n"
                "2. **ICICI Lombard SUV Plan** — ₹13,000–₹30,000/year\n"
                "   • Consumables cover (oils, filters)\n"
                "   • 24×7 roadside & towing assistance\n"
                "   • Spot claim settlement\n"
                "   Best for: Premium SUVs (Fortuner, Harrier, XUV)\n\n"
                "3. **Tata AIG** — ₹11,000–₹26,000/year\n"
                "   • Return to Invoice cover\n"
                "   • Key replacement add-on\n"
                "   • High NCB retention\n"
                "   Best for: Older SUVs with high mileage"
            )
        else:
            reply = (
                f"Based on your {vehicle_type}, here are the best insurance plans in India (2025):\n\n"
                "1. **HDFC ERGO** — ₹6,000–₹15,000/year\n"
                "   • Cashless repairs at 6,800+ garages\n"
                "   • Zero depreciation add-on available\n"
                "   • Quick digital claims\n"
                "   Best for: New & well-maintained cars\n\n"
                "2. **Bajaj Allianz** — ₹5,500–₹14,000/year\n"
                "   • 24×7 roadside assistance\n"
                "   • High claim settlement ratio (98%+)\n"
                "   • Engine protect add-on\n"
                "   Best for: Budget-conscious owners\n\n"
                "3. **ICICI Lombard** — ₹6,500–₹16,000/year\n"
                "   • 15,000+ network garages\n"
                "   • Instant policy issuance\n"
                "   • No-claim bonus up to 50%\n"
                "   Best for: Premium & older vehicles"
            )
        if has_damage:
            reply += "\n\n**Note:** Visible damage detected — consider adding a **Return to Invoice** or **Repair of Glass/Fibre** add-on to any plan above."
        if is_old:
            reply += "\n\n🔧 **Note:** Older vehicle detected — an **Engine Protection** add-on is strongly recommended."

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