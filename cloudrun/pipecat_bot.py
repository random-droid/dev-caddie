import argparse
import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
try:
    from pipecat.processors.frame_processor import FrameProcessor
except Exception:
    FrameProcessor = None
try:
    from pipecat.frames.frames import (
        StartInterruptionFrame,
        CancelFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    )
except Exception:
    StartInterruptionFrame = None
    CancelFrame = None
    BotStartedSpeakingFrame = None
    BotStoppedSpeakingFrame = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("briefing-bot")


SYSTEM_INSTRUCTION = (
    "You are a professional briefing assistant. Be concise and wait for the user "
    "to finish speaking. Use tools for real-time data when asked. "
    "Start the session by reading the top 5 items below as a morning briefing."
)


async def _fetch_top_daily_reads() -> List[Dict[str, Any]]:
    base_url = os.getenv("CONTENT_API_BASE", "http://localhost:8080").rstrip("/")
    logger.info("CONTENT_API_BASE=%s", base_url)
    if base_url.startswith("http://localhost"):
        logger.warning("CONTENT_API_BASE is localhost; set it to the public service URL in Cloud Run.")
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(f"{base_url}/api/briefing/top")
        res.raise_for_status()
        items = res.json().get("items", [])
    return [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "summary": item.get("summary") or item.get("description"),
        }
        for item in items[:5]
    ]


async def _fetch_articles(topic: str) -> List[Dict[str, Any]]:
    base_url = os.getenv("CONTENT_API_BASE", "http://localhost:8080").rstrip("/")
    logger.info("CONTENT_API_BASE=%s", base_url)
    if base_url.startswith("http://localhost"):
        logger.warning("CONTENT_API_BASE is localhost; set it to the public service URL in Cloud Run.")
    limit = int(os.getenv("BRIEFING_TOOL_LIMIT", "20"))
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(f"{base_url}/api/feeds/daily_reads", params={"limit": limit})
        res.raise_for_status()
        items = res.json().get("items", [])

    topic_lower = topic.lower()
    filtered = [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "summary": item.get("summary") or item.get("description"),
        }
        for item in items
        if topic_lower in (item.get("title", "").lower())
        or topic_lower in (item.get("summary", "").lower())
    ]
    return filtered[:5]


async def get_briefing_data(params: FunctionCallParams):
    topic = params.arguments.get("topic", "general")
    try:
        articles = await _fetch_articles(topic)
        await params.result_callback(
            {
                "status": "success",
                "topic": topic,
                "articles": articles,
            }
        )
    except Exception as exc:
        logger.exception("Tool error")
        await params.result_callback({"status": "error", "message": str(exc)})


def _build_tools():
    return [
        {
            "function_declarations": [
                {
                    "name": "get_briefing_data",
                    "description": "Fetches the latest LLM-related articles by topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Category or keyword (e.g., llm, agents, inference).",
                            }
                        },
                        "required": ["topic"],
                    },
                }
            ]
        }
    ]


async def main(room_url: str, token: str):
    logger.info("Starting Pipecat bot for room=%s", room_url)
    transport = DailyTransport(
        room_url,
        token,
        "Briefing Assistant",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    top_reads = []
    try:
        top_reads = await _fetch_top_daily_reads()
    except Exception:
        logger.exception("Failed to fetch top daily reads")

    if top_reads:
        formatted = "\n".join(
            [f"{i+1}. {item['title']} — {item.get('summary','')}" for i, item in enumerate(top_reads)]
        )
        system_instruction = f"{SYSTEM_INSTRUCTION}\n\nTop 5 daily reads:\n{formatted}"
        briefing_text = f"Please read today's briefing:\n{formatted}"
    else:
        logger.warning("No top reads found; proceeding without briefing list.")
        system_instruction = SYSTEM_INSTRUCTION
        briefing_text = "Please greet the user and start the daily briefing."

    llm = GeminiLiveVertexLLMService(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location="us-central1",
        model="google/gemini-live-2.5-flash-native-audio",
        voice_id="Puck",
        tools=_build_tools(),
        system_instruction=system_instruction,
    )
    llm.register_function("get_briefing_data", get_briefing_data)

    ui_sync_processor = None
    if FrameProcessor:
        class UiSyncProcessor(FrameProcessor):
            _start = {"BotStartedSpeakingFrame", "LLMResponseStartFrame", "BotSpeakingFrame"}
            _stop = {"BotStoppedSpeakingFrame", "LLMResponseEndFrame", "LLMResponseInterruptedFrame"}

            def __init__(self, transport):
                super().__init__()
                self._transport = transport

            async def _send_state(self, speaking: bool):
                payload = {"type": "bot-speaking", "value": speaking}
                try:
                    if hasattr(self._transport, "app_message"):
                        await self._transport.app_message(payload)
                    elif hasattr(self._transport, "send_app_message"):
                        await self._transport.send_app_message(payload)
                except Exception:
                    logger.debug("UI sync: app_message failed", exc_info=True)

            async def process_frame(self, frame, direction):
                if BotStartedSpeakingFrame and BotStoppedSpeakingFrame:
                    if isinstance(frame, BotStartedSpeakingFrame):
                        await self._send_state(True)
                    elif isinstance(frame, BotStoppedSpeakingFrame):
                        await self._send_state(False)
                else:
                    name = frame.__class__.__name__
                    if name in self._start:
                        await self._send_state(True)
                    elif name in self._stop:
                        await self._send_state(False)
                await self.push_frame(frame, direction)

        ui_sync_processor = UiSyncProcessor(transport)
    else:
        logger.warning("UI sync disabled: FrameProcessor unavailable in this Pipecat version.")

    if ui_sync_processor:
        pipeline = Pipeline([transport.input(), llm, ui_sync_processor, transport.output()])
    else:
        pipeline = Pipeline([transport.input(), llm, transport.output()])
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(_transport, _participant):
        logger.info("First participant joined; waiting for WebRTC stabilization...")
        await asyncio.sleep(2)
        logger.info("WebRTC stabilized. Triggering Gemini briefing...")
        await task.queue_frame(TextFrame(briefing_text))

    @transport.event_handler("on_audio_data")
    async def on_audio_data(_transport, _data):
        logger.debug("Bot is streaming audio bytes.")

    async def _handle_app_message(*args, **kwargs):
        payload = None
        for arg in args:
            if isinstance(arg, dict) and "type" in arg:
                payload = arg
        if payload is None and isinstance(kwargs.get("data"), dict):
            payload = kwargs.get("data")
        if payload and payload.get("type") == "user-interruption":
            logger.info("Frontend interruption signal received.")
            if StartInterruptionFrame and CancelFrame:
                await task.queue_frame(StartInterruptionFrame())
                await task.queue_frame(CancelFrame())

    @transport.event_handler("on_app_message")
    async def on_app_message(*args, **kwargs):
        await _handle_app_message(*args, **kwargs)

    @transport.event_handler("on_participant_left")
    async def on_participant_left(_transport, _participant, _reason):
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Gemini briefing bot")
    parser.add_argument("-u", "--url", required=True, help="Daily room URL")
    parser.add_argument("-t", "--token", required=True, help="Daily meeting token")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.token))
