import React, { useState, useEffect } from 'react';

export default function Whisper() {
    const [isRecording, setIsRecording] = useState(false);
    const [transcription, setTranscription] = useState('');

    useEffect(() => {
        // Load the Whisper model and other necessary setups
        async function loadWhisperModel() {
            // Assuming `loadWhisper` and other necessary functions are defined globally
            // or imported from a separate JS file included in your HTML.
            // You need to adapt this to your project setup.
            window.loadWhisper('tiny.en'); // Example model loading
        }

        loadWhisperModel();

        // Cleanup function to stop any ongoing recording
        return () => {
            window.onStop(); // Assuming this function stops the recording
        };
    }, []);

    const handleStartRecording = () => {
        window.onStart(); // Start recording
        setIsRecording(true);
    };

    const handleStopRecording = () => {
        window.onStop(); // Stop recording
        setIsRecording(false);
    };

    // Listen to transcription updates
    // This could be done via setting up a callback or event listener to update `setTranscription`
    // depending on how the Whisper JS code is structured.

    return (
        <div>
            <button onClick={handleStartRecording} disabled={isRecording}>Start</button>
            <button onClick={handleStopRecording} disabled={!isRecording}>Stop</button>
            <div>Transcription: {transcription}</div>
        </div>
    );
}
