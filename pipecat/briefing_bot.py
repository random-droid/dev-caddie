import os

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport


SYSTEM_INSTRUCTION = (
    "You are an energetic, friendly tech news anchor for a morning briefing. "
    "Read the briefing clearly and concisely. If the user interrupts with a question, "
    "pause reading, answer, and then ask if you should continue."
)


async def bot(runner_args: RunnerArguments):
    transport = DailyTransport(
        runner_args.room_url,
        runner_args.token,
        "BriefingBot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        ),
    )

    llm = GeminiLiveVertexLLMService(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location="us-central1",
        model="google/gemini-live-2.5-flash-native-audio",
        voice_id="Puck",
        system_instruction=SYSTEM_INSTRUCTION,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            llm,
            transport.output(),
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
