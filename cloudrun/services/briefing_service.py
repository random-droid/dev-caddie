import os
import json
import logging
import datetime
import asyncio
import base64
from array import array
import time
from typing import List, Dict, Any, Optional

# Third-party
from google import genai
from google.genai import types
from google.cloud import bigquery
import websockets
import vertexai

logger = logging.getLogger(__name__)

class BriefingService:
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id or os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location  # Default (Global)
        self.live_location = "us-central1" # Dedicated location for Live API
        self._genai_client = None
        self._bq_client = None

    @property
    def genai(self):
        if not self._genai_client:
            # Override project_id from environment variables if available
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", self.project_id)

            # Initialize Vertex AI for Text/Flash models
            vertexai.init(project=self.project_id, location=self.location)

            # Initialize genai client, it will now use the Vertex AI context
            self._genai_client = genai.Client(vertexai=True)
        return self._genai_client

    @property
    def bq(self):
        if not self._bq_client:
            self._bq_client = bigquery.Client(project=self.project_id)
        return self._bq_client

    async def get_todays_script(self) -> Optional[str]:
        """
        Fetches the pre-generated script from BigQuery for today.
        """
        query = f"""
            SELECT script_content 
            FROM `{self.project_id}.content_intelligence.daily_briefings` 
            WHERE date = CURRENT_DATE("America/Denver")
            ORDER BY date DESC
            LIMIT 1
        """
        try:
            query_job = self.bq.query(query)
            row = next(query_job.result(), None)
            if row:
                return row.script_content
        except Exception as e:
            logger.warning(f"Failed to fetch script from BQ: {e}")
        
        # If missing, generate on demand (Airflow handles BQ persistence)
        try:
            articles = await self.get_recent_top_articles(limit=5)
            if not articles:
                return None
            script = await self.generate_fallback_script(articles)
            # Note: BQ write moved to Airflow DAG (Cloud Run SA lacks permission)
            return script
        except Exception as e:
            logger.error(f"Failed to generate briefing script: {e}")
            return None

    async def _upsert_today_script(self, script: str) -> None:
        """
        Upsert today's script into BigQuery (date, script_content).
        """
        if not script:
            return
        try:
            tz = ZoneInfo("America/Denver")
        except Exception:
            tz = datetime.timezone(datetime.timedelta(hours=-7))
        today = datetime.datetime.now(tz).date().isoformat()

        query = f"""
        MERGE `{self.project_id}.content_intelligence.daily_briefings` T
        USING (SELECT @date AS date, @script AS script_content) S
        ON T.date = S.date
        WHEN MATCHED THEN
          UPDATE SET script_content = S.script_content
        WHEN NOT MATCHED THEN
          INSERT (date, script_content) VALUES (S.date, S.script_content)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "DATE", today),
                bigquery.ScalarQueryParameter("script", "STRING", script),
            ]
        )
        try:
            self.bq.query(query, job_config=job_config).result()
            logger.info("Stored daily briefing in BigQuery for %s", today)
        except Exception as e:
            logger.warning(f"Failed to write script to BQ: {e}")

    async def get_latest_script(self) -> Optional[Dict[str, Any]]:
        """
        Fetches the most recent script available from BigQuery.
        """
        query = f"""
            SELECT script_content, date
            FROM `{self.project_id}.content_intelligence.daily_briefings`
            ORDER BY date DESC
            LIMIT 1
        """
        try:
            query_job = self.bq.query(query)
            row = next(query_job.result(), None)
            if row:
                return {"script": row.script_content, "date": row.date}
        except Exception as e:
            logger.warning(f"Failed to fetch latest script from BQ: {e}")
        return None

    async def get_recent_top_articles(self, limit: int = 5) -> List[Dict]:
        """
        Fetches the top scored articles with freshness preference.
        Primary: last 24 hours. Fallback: last 7 days.
        """
        def _run_query(days: int) -> List[Dict]:
            query = f"""
                SELECT title, summary, key_topics, url
                FROM `{self.project_id}.content_intelligence.articles_scored`
                WHERE scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
                AND final_score >= 50
                ORDER BY final_score DESC
                LIMIT {limit}
            """
            query_job = self.bq.query(query)
            return [
                {"title": row.title, "summary": row.summary, "topics": row.key_topics}
                for row in query_job.result()
            ]
        try:
            articles = _run_query(1)
            if len(articles) >= limit:
                return articles
            fallback = _run_query(7)
            return fallback
        except Exception as e:
            logger.error(f"Failed to fetch content for briefing: {e}")
            return []

    async def generate_fallback_script(self, articles: List[Dict]) -> str:
        """
        Fallback: Generates a script on-the-fly if the DAG failed.
        Uses Gemini 1.5 Pro (or Gemini 3 if available).
        """
        # Prepare context from articles
        context = "\n\n".join([
            f"Title: {a.get('title')}\nSummary: {a.get('summary')}"
            for a in articles
        ])

        prompt = f"""
        You are a Tech News Anchor.
        Task: Write a quick 3-minute morning briefing script based on these 5 stories.
        
        Constraints:
        1. Total Length: ~2-3 minutes spoken (approx 300-450 words).
        2. Per Article: 2-3 sentences max. Focus on the key takeaway/actionable insight.
        3. Style: Energetic, professional, concise.
        4. **Filter**: If an article appears spiteful, is a rant, or has no substance (e.g. just a link), **SKIP IT** and choose another if possible, or mention it only briefly as a "community discussion" without amplifying the negativity.
        
        Structure:
        "Good morning! Here are your top 5 tech reads for today.
        
        First up: [Title]. [Summary]. Key takeaway: [Insight].
        
        Next: [Title]...
        
        ...
        
        That's your briefing. Want me to dive deeper into any of these?"
        
        Stories:
        {context}
        
        Output: Just the spoken text.
        """
        
        try:
            # Using 1.5 Pro as safe default, can swap to gemini-3-pro-001
            resp = await asyncio.to_thread(
                self.genai.models.generate_content,
                model="gemini-2.5-flash", 
                contents=prompt
            )
            return resp.text
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return "Good morning. I am unable to generate the briefing right now. Please check the feed manually."

    async def handle_websocket(self, websocket):
        """
        Manages the WebSocket connection for Gemini Live.
        Flow:
        1. Receive 'init' from Client.
        2. Fetch/Generate Script.
        3. Connect to Vertex AI Live API (BidiStream).
        4. Proxy Audio/Text bidirectionally between Client <-> Vertex.
        """
        try:
            # 1. Wait for Init
            init_msg = await websocket.receive_json()
            if not isinstance(init_msg, dict) or init_msg.get("command") not in {"init", "start"}:
                logger.warning(f"WS Init invalid: {init_msg}")
                await websocket.send_json({"type": "error", "content": "Invalid init payload"})
                return
            logger.info(f"WS Init: {init_msg}")

            # 2. Get Script (Context)
            script = init_msg.get("script")
            if not script:
                script = await self.get_todays_script()
            if not script:
                logger.info("No script found for today. Using default script.")
                script = "No significant updates today."

            # Send Script Preview to Client
            await websocket.send_json({"type": "script", "content": script})

            # 3. Setup Gemini Live Session (Interactive)
            logger.info("Configuring Gemini Live interactive session...")

            system_prompt = """You are an energetic, friendly tech news anchor for a morning briefing.

Your PRIMARY task is to read the script below. However, this is an INTERACTIVE session:
- If the user interrupts with a question, PAUSE reading and answer their question conversationally.
- After answering, ask "Shall I continue with the briefing?" and wait for confirmation.
- If they say yes, resume reading from where you left off.
- If they ask follow-up questions, answer those too.
- Be helpful, concise, and maintain your energetic anchor personality.

SCRIPT TO READ:
""" + script

            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                system_instruction=types.Content(parts=[types.Part.from_text(text=system_prompt)]),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                    )
                )
            )

            # Connect to Gemini Live 2.5 Flash Native Audio
            # Valid Model ID: gemini-live-2.5-flash-native-audio (US-Central1)
            logger.info(f"Attempting to connect to gemini-live-2.5-flash-native-audio in {self.live_location}...")
            
            # Explicitly create a live-capable client in the correct region
            live_client = genai.Client(vertexai=True, project=self.project_id, location=self.live_location)
            
            async with live_client.aio.live.connect(model="gemini-live-2.5-flash-native-audio", config=config) as session:
                logger.info("Connected to Gemini Live! Sending status...")
                await websocket.send_json({"type": "status", "content": "connected"})

                # Start command
                await session.send(input="Please read the entire script now, from the beginning to the very end. Do not skip any part.", end_of_turn=True)

                # Bidirectional proxy loop
                logger.info("Starting bidirectional proxy loop...")
                chunk_count = 0
                total_bytes = 0
                last_client_bytes_log = time.monotonic()
                client_bytes_since_log = 0
                user_speaking = False
                gemini_speaking = False
                shield_until = 0.0
                last_continue_at = 0.0
                min_speech_seconds = 0.2
                min_silence_seconds = 0.8
                min_continue_gap_seconds = 1.5
                speech_seconds = 0.0
                vad_rms_threshold = 0.005
                vad_sample_rate = 16000
                vad_silence_seconds = 0.0

                def _pcm16_rms(pcm_bytes: bytes) -> float:
                    if not pcm_bytes:
                        return 0.0
                    samples = array('h')
                    samples.frombytes(pcm_bytes)
                    if not samples:
                        return 0.0
                    sum_squares = 0.0
                    for s in samples:
                        fs = s / 32768.0
                        sum_squares += fs * fs
                    return (sum_squares / len(samples)) ** 0.5

                async def send_continue():
                    nonlocal last_continue_at
                    now = time.monotonic()
                    if now - last_continue_at < min_continue_gap_seconds:
                        return
                    last_continue_at = now
                    await session.send(
                        input=types.LiveClientContent(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text="Please continue with the briefing.")]
                                )
                            ],
                            turnComplete=True
                        )
                    )
                    await websocket.send_json({"type": "status", "content": "processing"})

                async def handle_audio_bytes(audio_bytes: bytes):
                    nonlocal user_speaking, vad_silence_seconds, shield_until, speech_seconds
                    if not audio_bytes:
                        return
                    if time.monotonic() < shield_until:
                        vad_silence_seconds = 0.0
                        return
                    frame_seconds = (len(audio_bytes) / 2) / vad_sample_rate
                    rms = _pcm16_rms(audio_bytes)
                    if rms >= vad_rms_threshold:
                        vad_silence_seconds = 0.0
                        speech_seconds += frame_seconds
                        if not user_speaking and speech_seconds >= min_speech_seconds:
                            user_speaking = True
                            await websocket.send_json({"type": "status", "content": "listening"})
                    elif user_speaking:
                        vad_silence_seconds += frame_seconds
                        if vad_silence_seconds >= min_silence_seconds:
                            user_speaking = False
                            vad_silence_seconds = 0.0
                            speech_seconds = 0.0
                            logger.info("Server VAD: end-of-speech detected; sending resume nudge (turn_complete).")
                            await send_continue()
                    else:
                        vad_silence_seconds = 0.0
                        speech_seconds = 0.0

                    await session.send(
                        input=types.LiveClientRealtimeInput(
                            mediaChunks=[
                                types.Blob(mime_type="audio/pcm;rate=16000", data=audio_bytes)
                            ]
                        )
                    )

                async def receive_from_gemini():
                    """Receive audio/text from Gemini and forward to client."""
                    nonlocal chunk_count, total_bytes, gemini_speaking, shield_until
                    logger.info(">>> ENTERING receive_from_gemini loop")
                    try:
                        async for response in session.receive():
                            # If response has audio data, send to Client
                            if response.data:
                                if not gemini_speaking:
                                    gemini_speaking = True
                                    shield_until = time.monotonic() + 1.5
                                chunk_count += 1
                                total_bytes += len(response.data)
                                client_bytes_since_log += len(response.data)
                                now = time.monotonic()
                                if now - last_client_bytes_log >= 1.0:
                                    logger.info(f"Audio out to client: {client_bytes_since_log} bytes/sec")
                                    client_bytes_since_log = 0
                                    last_client_bytes_log = now
                                if chunk_count % 50 == 0:
                                    logger.info(f"Audio chunk #{chunk_count}: {len(response.data)} bytes (total: {total_bytes})")
                                
                                # Send Raw PCM Audio (Binary Frame)
                                # Client expects ArrayBuffer = PCM16
                                await websocket.send_bytes(response.data)

                            if response.text:
                                logger.info(f"Text received: {response.text[:100]}...")
                                await websocket.send_json({"type": "caption", "content": response.text})

                            # Check for turn/generation complete
                            # Check for turn/generation complete and INTERRUPTION
                            if response.server_content:
                                sc = response.server_content
                                
                                # CRITICAL: Forward interruption signal so client stops audio
                                if getattr(sc, 'interrupted', False):
                                    logger.warning(">>> BARGE-IN DETECTED: Gemini native VAD reports interruption <<<")
                                    logger.info("Gemini native VAD: user interruption detected; stop local playback.")
                                    await websocket.send_json({"type": "interrupt"})

                                if getattr(sc, 'generation_complete', False):
                                    logger.info("Gemini turn complete: model finished speaking; awaiting user input.")
                                    # Do NOT return here. Keep the loop alive for multi-turn.
                                    await websocket.send_json({"type": "turn_complete"})
                                    # Signal frontend to show Resume button
                                    await websocket.send_json({"type": "status", "content": "complete"})
                                    gemini_speaking = False
                                    continue
                                    
                                if getattr(sc, 'turn_complete', False):
                                    logger.info("Gemini turn complete: model finished speaking; awaiting user input.")
                                    await websocket.send_json({"type": "turn_complete"})
                                    # Signal frontend to show Resume button
                                    await websocket.send_json({"type": "status", "content": "complete"})
                                    gemini_speaking = False
                                    # If turn_complete is true, but generation_complete was not,
                                    # we still want to continue the loop to receive more responses.
                                    continue

                    except Exception as e:
                        logger.error(f"Gemini receive error: {e}")
                    finally:
                        logger.warning("<<< receive_from_gemini loop ended (stream closed or interrupted) >>>")

                async def receive_from_client():
                    """Receive audio/commands from client and forward to Gemini."""
                    nonlocal user_speaking
                    try:
                        while True:
                            msg = await websocket.receive()

                            if msg['type'] == 'websocket.disconnect':
                                logger.info("Client disconnected")
                                break

                            # Handle JSON messages (Commands & Audio)
                            if 'text' in msg and msg['text']:
                                try:
                                    data = json.loads(msg['text'])

                                    # 1. Handle Audio Data (realtimeInput)
                                    if 'realtimeInput' in data:
                                        chunks = data['realtimeInput'].get('mediaChunks', [])
                                        for chunk in chunks:
                                            if 'data' in chunk:
                                                # Create Media Chunk from Base64
                                                # Gemini Client expects just the Blob
                                                audio_bytes = base64.b64decode(chunk['data'])

                                                if len(audio_bytes) > 0:
                                                    await handle_audio_bytes(audio_bytes)

                                    elif 'realtime_input' in data and data['realtime_input'].get('activity_signal'):
                                        activity = data['realtime_input']['activity_signal']
                                        if activity == "START_OF_ACTIVITY":
                                            realtime_input = types.LiveClientRealtimeInput(
                                                activityStart=types.ActivityStart()
                                            )
                                        elif activity == "END_OF_ACTIVITY":
                                            realtime_input = types.LiveClientRealtimeInput(
                                                activityEnd=types.ActivityEnd()
                                            )
                                        else:
                                            realtime_input = None

                                        if realtime_input:
                                            await session.send(input=realtime_input)

                                    # 2. Handle Commands
                                    elif data.get('command') == 'user_speaking':
                                        user_speaking = data.get('active', False)
                                        logger.info(f"COMMAND RECEIVED: user_speaking={user_speaking}")

                                        if user_speaking:
                                            # User started speaking (UI hint only)
                                            await websocket.send_json({"type": "status", "content": "listening"})
                                        else:
                                            # Ignore client-driven end-of-speech; server VAD controls nudges
                                            logger.info("Ignoring client end-of-speech; server VAD drives turn completion.")
                                    elif data.get('command') == 'resume':
                                        logger.info("COMMAND RECEIVED: resume")
                                        await send_continue()
                                    elif data.get('command') in {'init', 'start'}:
                                        logger.info(f"COMMAND RECEIVED: {data.get('command')}")

                                except Exception as e:
                                    logger.error(f"Error processing JSON message: {e}")

                            # Primary: Handle raw binary audio from AudioWorklet
                            if 'bytes' in msg and msg['bytes']:
                                await handle_audio_bytes(msg['bytes'])

                    except Exception as e:
                        if "disconnect" not in str(e).lower():
                            logger.error(f"Client receive error: {e}")

                # Run both receive loops concurrently
                gemini_task = asyncio.create_task(receive_from_gemini())
                client_task = asyncio.create_task(receive_from_client())

                # Keep the socket alive until the client disconnects.
                # If Gemini finishes early, keep listening for resume/next turns.
                done, pending = await asyncio.wait(
                    [client_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                if gemini_task.done():
                    logger.info("Gemini stream ended; keeping WS open for client.")
                    try:
                        exc = gemini_task.exception()
                        if exc:
                            logger.error(f"Gemini Task Exception: {exc}")
                    except:
                        pass

                for task in done:
                    if task == client_task:
                        logger.info("Socket closure triggered by: CLIENT_TASK (User disconnected)")
                        try:
                            exc = task.exception()
                            if exc:
                                logger.error(f"Client Task Exception: {exc}")
                        except:
                            pass

                # Cancel gemini task if still running
                if not gemini_task.done():
                    gemini_task.cancel()
                    try:
                        await gemini_task
                    except asyncio.CancelledError:
                        pass

                logger.info(f"Session ended. Total chunks: {chunk_count}, Total bytes: {total_bytes}")
                await websocket.send_json({"type": "status", "content": "complete"})

        except Exception as e:
            logger.error(f"WebSocket Error: {e}")
        finally:
            try:
                await websocket.send_json({"type": "status", "content": "complete"})
            except:
                pass
            try:
                await websocket.close()
            except:
                pass
