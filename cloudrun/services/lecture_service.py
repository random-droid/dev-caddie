import os
import hashlib
import time
import json
import logging
import datetime
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# Third-party imports
import yt_dlp
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.cloud import storage
from google.api_core.exceptions import NotFound

# Project imports
# Project imports
from .url_normalizer import normalize_url, generate_article_id
# Verified import fix (cloudrun package issue resolved)

logger = logging.getLogger(__name__)

class LectureService:
    def __init__(self, project_id: str, location: str = "us-central1", genai_project_id: Optional[str]=None):
        self.project_id = project_id
        self.genai_location = location
        self.infra_location = "us-central1" # Infrastructure (BQ, GCS) is strictly us-central1
        self.genai_project_id = genai_project_id or project_id
        
        self.dataset_id = "content_intelligence"
        self.table_id = "lecture_notes"
        
        # Prefer Env Var -> Then Computed Default
        default_bucket = f"content-intelligence-media-{self.project_id}"
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", default_bucket)
        
        # Lazy initialization
        self._bq_client = None
        self._storage_client = None
        self._genai_client = None

    @property
    def bq(self):
        if not self._bq_client:
            self._bq_client = self._setup_bigquery()
        return self._bq_client

    @property
    def storage(self):
        if not self._storage_client:
            self._storage_client = self._setup_gcs()
        return self._storage_client

    @property
    def genai(self):
        if not self._genai_client:
            self._genai_client = genai.Client(
                vertexai=True, 
                project=self.genai_project_id, 
                location=self.genai_location
            )
        return self._genai_client

    def _setup_bigquery(self) -> bigquery.Client:
        # Let BQ client auto-detect location. Forcing 'us-central1' breaks if dataset is 'US' multi-region.
        client = bigquery.Client(project=self.project_id)
        dataset_ref = client.dataset(self.dataset_id)
        try:
            client.get_dataset(dataset_ref)
        except NotFound:
            try:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = self.infra_location
                client.create_dataset(dataset)
            except Exception as e:
                logger.warning(f"Could not create dataset (might exist/perm issue). {e}")
            
        table_ref = dataset_ref.table(self.table_id)
        schema = [
            bigquery.SchemaField("article_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("original_url", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("summary", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("content", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("video_uri", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("transcript_segments", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("start", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("end", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("text", "STRING", mode="NULLABLE"),
            ]),
        ]
        try:
            table = client.get_table(table_ref)
            # Schema Migration: Check if new columns exist, if not add them
            existing_fields = {f.name for f in table.schema}
            if "transcript_segments" not in existing_fields:
                logger.info("Migrating Schema: Adding transcript_segments column")
                new_schema = table.schema[:]
                new_schema.append(
                    bigquery.SchemaField("transcript_segments", "RECORD", mode="REPEATED", fields=[
                        bigquery.SchemaField("start", "FLOAT", mode="NULLABLE"),
                        bigquery.SchemaField("end", "FLOAT", mode="NULLABLE"),
                        bigquery.SchemaField("text", "STRING", mode="NULLABLE"),
                    ])
                )
                table.schema = new_schema
                client.update_table(table, ["schema"])
        except NotFound:
            try:
                table = bigquery.Table(table_ref, schema=schema)
                table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="created_at")
                client.create_table(table)
            except Exception as e:
                logger.warning(f"Could not create table. {e}")
        return client

    def _setup_gcs(self) -> "storage.Client":
        client = storage.Client(project=self.project_id)
        try:
            client.get_bucket(self.bucket_name)
        except NotFound:
            try:
                bucket = client.bucket(self.bucket_name)
                bucket.location = self.infra_location
                client.create_bucket(bucket)
            except Exception as e:
                logger.warning(f"Could not create bucket. {e}")
        return client

    def get_all_lectures(self) -> list[Dict[str, Any]]:
        """Retrieves list of lectures from BigQuery."""
        query = f"""
            SELECT article_id, title, summary, url, created_at 
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}` 
            ORDER BY created_at DESC
        """
        try:
            query_job = self.bq.query(query)
            results = []
            for row in query_job:
                results.append(dict(row))
            return results
        except Exception as e:
            logger.error(f"Failed to query lectures: {e}")
            return []

    def get_lecture_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single lecture by ID."""
        query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE article_id = @article_id
            ORDER BY created_at DESC
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("article_id", "STRING", article_id)
            ]
        )
        try:
            results = list(self.bq.query(query, job_config=job_config).result())
            if results:
                row = dict(results[0])
                return self._normalize_lecture_row(row)
            return None
        except Exception as e:
            logger.error(f"Failed to fetch lecture {article_id}: {e}")
            return None

    def _normalize_lecture_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        If older rows stored JSON in `content`, unpack it into content/summary/title
        so the lecture page renders correctly without reprocessing.
        """
        debug = {"path": "none", "ok": False}
        content = row.get("content")
        if isinstance(content, str):
            stripped = content.lstrip()
            if stripped.startswith("{") and '"markdown"' in stripped:
                # Try 1: Full JSON parse via _clean_json_response
                try:
                    cleaned = self._clean_json_response(content)
                    logger.debug(f"Cleaned JSON length: {len(cleaned)}, first 200 chars: {cleaned[:200]}")
                    data = json.loads(cleaned)
                    if isinstance(data, dict) and data.get("markdown"):
                        row["content"] = data["markdown"]
                        if data.get("summary") and not row.get("summary"):
                            row["summary"] = data["summary"]
                        if data.get("title") and (not row.get("title") or row.get("title") == "Unknown Title"):
                            row["title"] = data["title"]
                        debug = {"path": "json", "ok": True, "title_extracted": bool(data.get("title"))}
                        return row
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed at pos {e.pos}: {e.msg}. Context: ...{content[max(0,e.pos-50):e.pos+50]}...")
                except Exception as e:
                    logger.warning(f"JSON parse failed: {e}")

                # Try 2: Regex extraction for simple string fields (title, summary)
                def _extract_simple_field(field: str) -> Optional[str]:
                    # Match: "field": "value" (non-greedy, handles escapes)
                    pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*?)"'
                    match = re.search(pattern, content)
                    if match:
                        try:
                            return json.loads(f'"{match.group(1)}"')
                        except Exception:
                            pass
                    return None

                title = _extract_simple_field("title")
                if title and (not row.get("title") or row.get("title") == "Unknown Title"):
                    row["title"] = title
                summary = _extract_simple_field("summary")
                if summary and not row.get("summary"):
                    row["summary"] = summary

                # Try 3: Extract markdown field (can be very long with embedded newlines)
                # Find "markdown": " then scan for the closing quote (respecting escapes)
                md_match = re.search(r'"markdown"\s*:\s*"', content)
                if md_match:
                    start_idx = md_match.end()
                    # Walk the string to find the unescaped closing quote
                    i = start_idx
                    while i < len(content):
                        ch = content[i]
                        if ch == '\\' and i + 1 < len(content):
                            i += 2  # skip escaped char
                        elif ch == '"':
                            # Found closing quote
                            raw_literal = content[start_idx:i]
                            try:
                                row["content"] = json.loads(f'"{raw_literal}"')
                                debug = {"path": "regex", "ok": True}
                            except Exception as e2:
                                logger.warning(f"Markdown literal decode failed: {e2}")
                                # Last resort: manual unescape
                                unescaped = raw_literal.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
                                row["content"] = unescaped
                                debug = {"path": "manual", "ok": True}
                            break
                        else:
                            i += 1

        row["_debug_normalize"] = debug
        return row

    def delete_lecture(self, article_id: str) -> bool:
        """Deletes a lecture from BigQuery and GCS."""
        logger.info(f"Deleting lecture {article_id}...")
        try:
            # 1. BigQuery
            query = f"DELETE FROM `{self.project_id}.{self.dataset_id}.{self.table_id}` WHERE article_id = @article_id"
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("article_id", "STRING", article_id)
                ]
            )
            self.bq.query(query, job_config=job_config).result()
            logger.info("✅ Deleted from BigQuery.")
            
            # 2. GCS
            blobs = list(self.storage.bucket(self.bucket_name).list_blobs(prefix=f"{article_id}/"))
            if blobs:
                self.storage.bucket(self.bucket_name).delete_blobs(blobs)
                logger.info(f"✅ Deleted {len(blobs)} files from GCS.")
            else:
                logger.info("ℹ️  No GCS files found.")
                
            return True
        except Exception as e:
            logger.error(f"Failed to delete lecture {article_id}: {e}")
            return False

    def process_video(self, url: str, mock: bool = False, mode: str = "comprehensive", force: bool = False,
                       local_file: Optional[str] = None, skip_validation: bool = False,
                       max_slides: int = 0, min_interval: int = 0) -> Optional[str]:
        """
        Main workflow to process a video URL.

        Args:
            url: YouTube URL (used for article ID and timestamp links)
            mock: If True, returns mock data
            mode: "comprehensive" (Study Guide) or "transcript" (Verbatim)
            force: If True, regenerate even if exists
            local_file: If provided, skip download and use this file instead
            skip_validation: If True, skip Gemini vision validation of frames (faster)
            max_slides: Maximum slides to extract (0 = unlimited)
            min_interval: Minimum seconds between slides (0 = no minimum)
        """
        normalized_url = url.strip() # normalize_url(url) - Keep simple for now
        article_id = hashlib.md5(normalized_url.encode()).hexdigest()

        # Check if exists (unless forced)
        if not force:
            existing = self.get_lecture_by_id(article_id)
            if existing:
                logger.info(f"Lecture {article_id} already exists (Use force=True to overwrite)")
                return article_id

        logger.info(f"Processing {url} -> ID: {article_id}")

        # Download or use local file
        video_title = "Unknown Title"
        if local_file:
            if not os.path.exists(local_file):
                logger.error(f"Local file not found: {local_file}")
                return None
            video_path = local_file
            # Try to extract title from filename
            video_title = os.path.splitext(os.path.basename(local_file))[0]
            logger.info(f"Using local file: {local_file}")
        else:
            video_path, video_title = self._download_video(url)
            if not video_path: return None

        # Upload
        gcs_uri = self._upload_to_gcs(video_path)
        if not gcs_uri: return None
        
        # Generate (two-step: markdown + metadata)
        notes_bundle = self._generate_notes(gcs_uri, mock=mock, mode=mode)
        raw_response = json.dumps(notes_bundle) if notes_bundle else None

        # Post-Process (Extract Slides & Format Markdown)
        if notes_bundle and notes_bundle.get("markdown"):
            notes_json = json.dumps({
                "markdown": notes_bundle.get("markdown", ""),
                "slides": notes_bundle.get("slides", []),
            })
            # We need the local video path for ffmpeg
            notes = self._process_slides(video_path, notes_json, url, article_id=article_id, skip_validation=skip_validation)
        else:
            notes = None
        
        # Cleanup GCS (always) and local file (only if we downloaded it)
        self._cleanup_gcs(gcs_uri)
        if not local_file:
            # Only delete if we downloaded it (don't delete user's local file)
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception as e:
                logger.warning(f"Failed to remove local video {video_path}: {e}")
            
        # Save
        if notes:
            transcript_segments = []
            data = notes_bundle or {}

            # Title priority: YouTube title > Gemini JSON title > Markdown extraction
            # YouTube title is most reliable; Gemini often extracts section headers
            _, extracted_summary = self._extract_title_summary(notes)
            json_title = data.get("title") if isinstance(data, dict) else None

            # Prefer video_title (from yt-dlp) unless it's generic
            if video_title and video_title != "Unknown Title":
                final_title = video_title
            elif json_title and json_title != "Unknown Title":
                final_title = json_title
            else:
                final_title = "Lecture Notes"

            # Use JSON summary if available (Crucial fix for "No summary available")
            final_summary = data.get("summary") if isinstance(data, dict) and data.get("summary") else extracted_summary

            bq_data = {
                "article_id": article_id,
                "url": normalized_url,
                "original_url": url,
                "title": final_title,
                "summary": final_summary,
                "content": notes,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "transcript_segments": transcript_segments
            }
            self._save_to_bigquery(bq_data)
            return article_id

    def _download_video(self, url: str, output_dir: str = "/tmp/downloads") -> tuple[Optional[str], Optional[str]]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ydl_opts = {
            'format': '18/best',
            'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
            'quiet': False,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                video_id = info_dict.get("id", None)
                video_ext = info_dict.get("ext", None)
                video_title = info_dict.get("title", "Unknown Title")
                if video_id and video_ext:
                    return os.path.join(output_dir, f"{video_id}.{video_ext}"), video_title
                raise RuntimeError("yt_dlp extraction failed: Missing ID or Ext")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise RuntimeError(f"Video Download Failed: {e}")

    def _upload_to_gcs(self, video_path: str) -> Optional[str]:
        blob_name = os.path.basename(video_path)
        blob = self.storage.bucket(self.bucket_name).blob(blob_name)
        if not blob.exists():
            blob.upload_from_filename(video_path, timeout=600)
        return f"gs://{self.bucket_name}/{blob_name}"

    def _cleanup_gcs(self, gcs_uri: str):
        try:
            blob_name = "/".join(gcs_uri.split("/")[3:])
            self.storage.bucket(self.bucket_name).blob(blob_name).delete()
        except Exception as e:
            logger.warning(f"Cleanup failed for {gcs_uri}: {e}")

    def _generate_notes(self, gcs_uri: str, mock: bool = False, mode: str = "comprehensive") -> Optional[Dict[str, Any]]:
        """Generate lecture notes from video using Gemini.

        True two-pass flow:
        1) Generate clean markdown WITHOUT any slide markers.
        2) Second pass: Insert [SLIDE: MM:SS] markers ONLY at paragraph boundaries.
        3) Parse final markdown for slides/title/summary.
        """
        if mock:
            time.sleep(2)
            return {
                "title": "Mock Notes",
                "summary": "Mock summary",
                "markdown": "# Mock Notes\n\n#### [SLIDE: 00:05]\n\nMock content here.",
                "slides": [{"timestamp": "00:05", "description": "Intro"}],
            }

        if mode == "transcript":
            notes_prompt = """
            You are an expert transcriber.
            Task: Create a verbatim transcript of this video as Markdown.

            Output must be ONLY Markdown (no JSON, no code fences).
            - Include timestamps in [MM:SS] or [HH:MM:SS]
            - Do NOT include any [SLIDE:] markers - those will be added in a second pass.
            """
        else:
            # PASS 1: Generate clean notes WITHOUT slide markers
            notes_prompt = """
            You are an expert academic note-taker.
            Task: Create a Master Study Guide for this video as Markdown.

            Output must be ONLY Markdown (no JSON, no code fences).
            IMPORTANT: Do NOT include any [SLIDE:] markers. Those will be added separately.

            Markdown Structure (Strict Order):
            1.  **Quick Summary & Key Takeaways**
                *   Provide a "High-Level Summary" (3 sentences).
                *   Create a table of "Top 5 Key Takeaways" with columns: | Takeaway | Timestamp | Explanation |
            2.  **Comprehensive Lecture Notes**
                *   **Detailed Outline**: A hierarchical outline with timestamps [HH:MM:SS].
                *   **Visual Aids**: Describe diagrams/slides in detail where relevant.
                *   **Key Terms & Definitions** and **Key Examples**
            3.  **Cornell Study Notes**
                *   Table: | Keywords & Queries | Detailed Notes |
            """
        try:
            # Pass 1: Generate clean markdown
            logger.info("Pass 1: Generating clean notes (no slide markers)...")
            response = self.genai.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4"),
                            types.Part.from_text(text=notes_prompt),
                        ]
                    )
                ]
            )
            clean_markdown = response.text or ""
            logger.info(f"Pass 1 complete: {len(clean_markdown)} chars")

            # Pass 2: Insert slide markers at paragraph boundaries
            logger.info("Pass 2: Inserting slide markers at paragraph boundaries...")
            slide_insertion_prompt = """
            You are reviewing lecture notes alongside the original video.
            Task: Insert [SLIDE: MM:SS] markers where meaningful visual content appears.

            PLACEMENT RULES:
            - Insert exactly ONE "Hero Slide" under "Quick Summary & Key Takeaways" (most representative frame).
            - Insert ALL other slides ONLY within "Comprehensive Lecture Notes" section.
            - NEVER insert slides in "Cornell Study Notes" (tables only, no images).

            FORMATTING RULES:
            1. Output the COMPLETE notes with slide markers inserted.
            2. Format EVERY marker as a Level 4 Header on its OWN line: #### [SLIDE: MM:SS]
            3. Insert markers ONLY at the START of paragraphs - NEVER mid-sentence.
            4. ALWAYS have a blank line BEFORE and AFTER each slide header.
            5. Only mark timestamps when there is actual visual content (diagrams, slides, code).
            6. Do NOT mark when only a face/webcam or blank screen is visible.

            Example of correct placement:

            Some paragraph ends here.

            #### [SLIDE: 05:32]

            Next topic begins here with new visual content.

            Here are the notes to process:
            """

            # Note: Don't send video again in Pass 2 - timestamps are already in the notes.
            # This avoids doubling token count for long videos.
            slide_response = self.genai.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=slide_insertion_prompt),
                            types.Part.from_text(text=clean_markdown),
                        ]
                    )
                ]
            )
            markdown = slide_response.text or clean_markdown
            logger.info(f"Pass 2 complete: {len(markdown)} chars")

            # Pass 3: Parse slides/title/summary from final markdown
            parse_prompt = """
            Extract a JSON object from the following Markdown.
            Return ONLY valid JSON (no code fences).
            Schema:
            {
              "title": "string",
              "summary": "string",
              "slides": [ {"timestamp": "MM:SS", "description": "string"} ]
            }
            - title: use the H1 title if present, else first strong title.
            - summary: 2-3 sentence summary of the document.
            - slides: every [SLIDE: MM:SS] occurrence with a short description inferred from nearby text.
            """
            parse_response = self.genai.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(response_mime_type="application/json"),
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=parse_prompt),
                            types.Part.from_text(text=markdown),
                        ],
                    )
                ],
            )
            meta = {}
            try:
                cleaned = self._clean_json_response(parse_response.text or "")
                meta = json.loads(cleaned)
            except Exception as e:
                logger.warning(f"Failed to parse markdown metadata JSON: {e}")
                meta = {}

            return {
                "title": meta.get("title"),
                "summary": meta.get("summary"),
                "markdown": markdown,
                "slides": meta.get("slides") or [],
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"GenAI Content Generation Failed: {e}")

    def _timespec_to_seconds(self, timespec: str) -> int:
        # MM:SS or HH:MM:SS to seconds
        try:
            parts = timespec.split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                secs = int(float(parts[1]))
                return mins * 60 + secs
            if len(parts) == 3:
                hours = int(parts[0])
                mins = int(parts[1])
                secs = int(float(parts[2]))
                return hours * 3600 + mins * 60 + secs
            return 0
        except: return 0

    def _normalize_timespec(self, timespec: str) -> str:
        """Normalize a time string to MM:SS or HH:MM:SS (no fractions)."""
        try:
            total = self._timespec_to_seconds(timespec)
            if total <= 0:
                return timespec.strip()
            hours = total // 3600
            mins = (total % 3600) // 60
            secs = total % 60
            if hours > 0:
                return f"{hours:02d}:{mins:02d}:{secs:02d}"
            return f"{mins:02d}:{secs:02d}"
        except Exception:
            return timespec.strip()

    def _is_slide_worthy(self, image_path: str) -> bool:
        """Use Gemini Vision to check if frame contains useful visual content vs just a face."""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            response = self.genai.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text="""Analyze this video frame. Is it worth including in study notes?

KEEP (answer SLIDE) if it shows:
- A presentation slide with text/bullet points
- Code or terminal output
- A diagram, chart, or architecture drawing
- An infographic or whiteboard with content
- Any visual that conveys meaningful information

SKIP (answer SKIP) if it shows:
- A person's face or webcam view
- A blank or nearly blank screen
- A title-only slide with no content
- A transition or loading screen
- A video thumbnail or preview
- Mostly empty space with minimal text

Be strict: only SLIDE if there's substantial visual content worth capturing.

Answer with just one word: SLIDE or SKIP"""),
                        ]
                    )
                ]
            )
            result = response.text.strip().upper()
            is_worthy = "SLIDE" in result and "SKIP" not in result
            logger.info(f"Frame validation: {image_path} -> {result} (worthy={is_worthy})")
            return is_worthy
        except Exception as e:
            logger.warning(f"Frame validation failed for {image_path}: {e}. Keeping frame.")
            return True  # Keep frame on error (fail-open)

    def _clean_json_response(self, text: str) -> str:
        """
        Clean Gemini response to ensure valid JSON.
        Removes markdown code blocks (```json ... ```) and optional preambles.
        """
        text = text.strip()

        # 1) Try fenced code blocks first - but only if not already JSON
        if not text.startswith("{") and not text.startswith("["):
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
            if fence_match:
                text = fence_match.group(1).strip()

        # 2) If there is extra preamble, extract the first balanced JSON object/array
        def _extract_balanced_json(src: str) -> str:
            in_str = False
            escape = False
            depth = 0
            start = None
            for i, ch in enumerate(src):
                if escape:
                    escape = False
                    continue
                if in_str:
                    if ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                # Outside string
                if ch == '"':
                    in_str = True
                    continue
                if ch in "{[":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0 and start is not None:
                        return src[start:i + 1]
            return src

        text = _extract_balanced_json(text).strip()

        # 3) If the response still contains leading "json" on its own line, drop it
        if text.lower().startswith("json"):
            text = "\n".join(text.split("\n")[1:]).strip()

        return text

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to get video duration: {e}")
            return float('inf') # Assume infinite if we can't check, let ffmpeg handle it

    def _extract_frames(self, video_path: str, slides: list, validate: bool = True,
                         max_slides: int = 0, min_interval: int = 0) -> list:
        """Extract frames using ffmpeg, optionally validating with Gemini Vision.

        Args:
            video_path: Path to video file
            slides: List of slide dicts with 'timestamp' keys
            validate: Whether to use Gemini vision to validate frames
            max_slides: Maximum number of slides to extract (0 = unlimited)
            min_interval: Minimum seconds between slides (0 = no minimum)
        """
        output_paths = []
        skipped_count = 0

        duration = self._get_video_duration(video_path)
        logger.info(f"Video duration: {duration}s")

        # Deduplicate slides by seconds to avoid overwriting/deleting frames
        seen_seconds = set()
        unique_slides = []
        for slide in slides:
            ts = slide.get("timestamp", "00:00")
            sec = self._timespec_to_seconds(ts)
            if sec not in seen_seconds:
                seen_seconds.add(sec)
                unique_slides.append(slide)
            else:
                logger.debug(f"Skipping duplicate timestamp {ts} ({sec}s)")

        if len(unique_slides) < len(slides):
            logger.info(f"Deduplicated {len(slides)} slides to {len(unique_slides)} unique timestamps")

        # Apply minimum interval filter
        if min_interval > 0 and unique_slides:
            filtered_slides = []
            last_sec = -min_interval  # Allow first slide
            for slide in unique_slides:
                sec = self._timespec_to_seconds(slide.get("timestamp", "00:00"))
                if sec - last_sec >= min_interval:
                    filtered_slides.append(slide)
                    last_sec = sec
                else:
                    logger.debug(f"Skipping slide at {slide.get('timestamp')} - too close to previous ({sec - last_sec}s < {min_interval}s)")
            if len(filtered_slides) < len(unique_slides):
                logger.info(f"Interval filter: {len(unique_slides)} -> {len(filtered_slides)} slides (min {min_interval}s apart)")
            unique_slides = filtered_slides

        # Apply max slides limit (evenly distributed)
        if max_slides > 0 and len(unique_slides) > max_slides:
            # Sort by timestamp and pick evenly spaced slides
            unique_slides.sort(key=lambda s: self._timespec_to_seconds(s.get("timestamp", "00:00")))
            step = len(unique_slides) / max_slides
            selected = []
            for i in range(max_slides):
                idx = int(i * step)
                if idx < len(unique_slides):
                    selected.append(unique_slides[idx])
            logger.info(f"Max slides filter: {len(unique_slides)} -> {len(selected)} slides (max {max_slides})")
            unique_slides = selected

        for slide in unique_slides:
            ts = slide.get("timestamp", "00:00")
            sec = self._timespec_to_seconds(ts)
            
            if sec > duration - 1: # Buffer of 1s
                logger.warning(f"Skipping timestamp {ts} ({sec}s) as it exceeds video duration ({duration}s)")
                continue

            out_name = f"{video_path}_{sec}.jpg"
            # ffmpeg -ss <sec> -i <input> -frames:v 1 -q:v 2 <out>
            cmd = [
                "ffmpeg", "-ss", str(sec), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", "-y", out_name
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(out_name):
                    # Validate frame if enabled
                    if validate and not self._is_slide_worthy(out_name):
                        logger.info(f"Skipping low-value frame at {ts}")
                        os.remove(out_name)
                        skipped_count += 1
                        continue
                    output_paths.append({"path": out_name, "timestamp": ts})
            except subprocess.CalledProcessError as e:
                # 234 = Read past EOF usually
                logger.warning(f"FFmpeg failed to extract frame at {ts} (Exit {e.returncode}): {e}")
            except Exception as e:
                logger.warning(f"Generic error extracting frame at {ts}: {e}")

        if skipped_count > 0:
            logger.info(f"Filtered out {skipped_count} low-value frames, kept {len(output_paths)} quality slides")

        return output_paths

    def _process_slides(self, video_path: str, notes_json: str, video_url: str = "",
                         article_id: str = "", skip_validation: bool = False, max_slides: int = 0, min_interval: int = 0) -> str:
        try:
            # Clean and Parse
            cleaned_json = self._clean_json_response(notes_json)
            data = json.loads(cleaned_json)
            
            markdown = data.get("markdown", "")
            slides = data.get("slides", [])

            # Normalize slide placeholders in markdown to match slide timestamps
            def _normalize_placeholder(match):
                ts = match.group(1)
                return f"[SLIDE: {self._normalize_timespec(ts)}]"
            markdown = re.sub(r"\[SLIDE:\s*([0-9:.]+)\s*\]", _normalize_placeholder, markdown, flags=re.IGNORECASE)

            # Normalize slide timestamps to MM:SS / HH:MM:SS
            if slides:
                for slide in slides:
                    ts = slide.get("timestamp", "")
                    if ts:
                        slide["timestamp"] = self._normalize_timespec(ts)
            
            # 0. Link Timestamps first (before inserting images)
            markdown = self._link_timestamps(markdown, video_url)
            
            if not slides:
                logger.info("No slides in JSON response from Gemini")
                return markdown

            logger.info(f"Processing {len(slides)} slide timestamps from Gemini")

            # Extract
            frames = self._extract_frames(video_path, slides, validate=not skip_validation,
                                          max_slides=max_slides, min_interval=min_interval)

            if not frames:
                logger.warning("No frames passed validation - all were filtered out or extraction failed")
                return markdown

            logger.info(f"Uploading {len(frames)} validated frames to GCS")
            
            # Upload & Inject
            for frame in frames:
                # Upload
                blob_name = f"{article_id}/slides/{os.path.basename(frame['path'])}"
                blob = self.storage.bucket(self.bucket_name).blob(blob_name)
                blob.upload_from_filename(frame['path'])
                
                # Make Public for Hackathon Judges
                try:
                    blob.make_public()
                except Exception as e:
                    logger.warning(f"Could not make slide public (Project Policy?): {e}")

                public_url = blob.public_url
                
                # Cleanup local jpg
                os.remove(frame['path'])
                
                # Replace in Markdown using REGEX for robustness
                # Matches #### [SLIDE: 10:05] or just [SLIDE: 10:05] etc.
                ts = frame['timestamp']
                # Escape regex special chars if any (unlikely in MM:SS)
                # Include optional #### header prefix to avoid leaving empty headers
                pattern = r"(?:####\s*)?\[SLIDE:\s*" + re.escape(ts) + r"\s*\]"

                img_md = f"\n#### [Slide {ts}]\n![Slide {ts}]({public_url})\n*Slide: {ts}*\n"

                # Debug: Check if pattern exists in markdown
                if re.search(pattern, markdown, flags=re.IGNORECASE):
                    logger.info(f"Replacing [SLIDE: {ts}] with image")
                else:
                    logger.warning(f"Pattern not found for [SLIDE: {ts}] - checking markdown for similar...")
                    # Show what slide patterns ARE in the markdown
                    found = re.findall(r"\[SLIDE:\s*[^\]]+\]", markdown[:2000])
                    if found:
                        logger.warning(f"Found patterns in markdown: {found[:5]}")

                # Perform regex replace
                markdown = re.sub(pattern, img_md, markdown, flags=re.IGNORECASE)
                
            return markdown
        except Exception as e:
            logger.error(f"Slide processing failed: {e}")
            # return raw markdown if json parse works, else plain text
            if 'data' in locals(): return data.get("markdown", "")
            return str(notes_json)

    def _link_timestamps(self, markdown: str, video_url: str) -> str:
        """Converts [MM:SS] or [HH:MM:SS] to linked timestamps."""
        if not video_url: return markdown
        
        # Regex to find [D:DD] or [D:DD:DD]
        # We capture the inner time text
        pattern = r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]"
        
        def replacer(match):
            ts = match.group(1)
            seconds = self._timespec_to_seconds(ts)
            # YouTube format: &t=123s
            # Support various url formats? For now, standard youtube.
            # If standard URL: https://www.youtube.com/watch?v=ID
            # Link: https://www.youtube.com/watch?v=ID&t=123s
            if "?" in video_url:
                separator = "&"
            else:
                separator = "?"
                
            link = f"{video_url}{separator}t={seconds}s"
            return f"[{ts}]({link})"

        return re.sub(pattern, replacer, markdown)

    def _extract_title_summary(self, text: str):
        """Extract title and summary from markdown notes."""
        title = "Unknown Title"
        summary = ""
        lines = text.split('\n')

        # Extract title
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
            elif line.startswith('**Title'):
                title = line.split(':', 1)[1].strip()
                break

        # Extract summary: first non-empty paragraph after title
        in_content = False
        summary_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip title lines and empty lines at start
            if stripped.startswith('#') or stripped.startswith('**Title'):
                in_content = True
                continue
            if not in_content:
                continue
            # Skip section headers
            if stripped.startswith('#') or stripped.startswith('**'):
                if summary_lines:
                    break
                continue
            # Collect content lines
            if stripped:
                summary_lines.append(stripped)
                if len(summary_lines) >= 3:  # Get first ~3 lines for summary
                    break

        if summary_lines:
            summary = ' '.join(summary_lines)[:500]  # Limit to 500 chars

        return title, summary

    def _save_to_bigquery(self, data: Dict[str, Any]):
        errors = self.bq.insert_rows_json(self.bq.dataset(self.dataset_id).table(self.table_id), [data])
        if errors: logger.error(f"BQ Insert errors: {errors}")
