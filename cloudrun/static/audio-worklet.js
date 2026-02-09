class GeminiProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.targetSampleRate = 16000;
        this.buffer = new Int16Array(2048);
        this.bufferIndex = 0;
        this.sourceCursor = 0;

        // --- Silence Detection ---
        // RMS Threshold: 0.005 is a good baseline for 16-bit normalised audio
        this.silenceThreshold = 0.005;
        this.silenceDurationSamples = 0;
        this.maxSilenceSamples = 16000 * 0.8; // 800ms at 16kHz
        this.isSpeaking = false;
        this.speechDurationSamples = 0;
        this.minSpeechSamples = 16000 * 0.15; // 150ms of sustained speech
        this.hadSpeechStart = false;
    }

    process(inputs) {
        const inputData = inputs[0][0];
        if (!inputData) return true;

        // 1. Calculate RMS (Root Mean Square) for volume
        let sumSquares = 0;
        for (let i = 0; i < inputData.length; i++) {
            sumSquares += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sumSquares / inputData.length);

        // 2. Debug Logging (Random sampling to avoid console spam)
        if (rms > 0.001 && Math.random() < 0.05) {
            this.port.postMessage({ type: 'vad_debug', rms: rms });
        }

        // 3. VAD Logic (State Machine)
        if (rms > this.silenceThreshold) {
            // SPEECH DETECTED
            this.silenceDurationSamples = 0;
            this.speechDurationSamples += inputData.length * (this.targetSampleRate / sampleRate);
            if (!this.isSpeaking && this.speechDurationSamples >= this.minSpeechSamples) {
                this.isSpeaking = true;
                // Local UI hook only (do not signal Gemini)
                this.port.postMessage({ type: 'speech_start' });
                this.hadSpeechStart = true;
            }
        } else {
            // SILENCE DETECTED
            if (this.isSpeaking && this.hadSpeechStart) {
                // Accumulate silence duration
                this.silenceDurationSamples += inputData.length * (this.targetSampleRate / sampleRate);

                // If silence exceeds 800ms, trigger turn end
                if (this.silenceDurationSamples >= this.maxSilenceSamples) {
                    this.isSpeaking = false;
                    this.port.postMessage({ type: 'silence_detected' });
                    this.silenceDurationSamples = 0;
                    this.speechDurationSamples = 0;
                    this.hadSpeechStart = false;
                }
            } else {
                // Do not emit silence events until speech has started
                this.silenceDurationSamples = 0;
                this.speechDurationSamples = 0;
                this.hadSpeechStart = false;
            }
        }

        // 4. Audio Resampling & Buffering (Linear Interpolation)
        const ratio = sampleRate / this.targetSampleRate;

        while (this.sourceCursor < inputData.length) {
            const i = Math.floor(this.sourceCursor);
            const nextI = Math.min(i + 1, inputData.length - 1);
            const weight = this.sourceCursor - i;

            let sample = inputData[i] * (1 - weight) + inputData[nextI] * weight;
            sample = Math.max(-1, Math.min(1, sample)); // Clip

            // Convert float (-1.0 to 1.0) to Int16
            this.buffer[this.bufferIndex++] = sample * 0x7FFF;

            if (this.bufferIndex >= this.buffer.length) {
                // Send full buffer to main thread
                this.port.postMessage({ type: 'audio', data: this.buffer.buffer.slice(0) });
                this.bufferIndex = 0;
            }
            this.sourceCursor += ratio;
        }

        this.sourceCursor -= inputData.length;
        return true;
    }
}

registerProcessor('gemini-processor', GeminiProcessor);
