import React, { useState, useRef, useEffect } from 'react';
import { pipeline } from '@xenova/transformers';
import { PCA } from 'ml-pca';

const generator = await pipeline('text-generation', 'Xenova/Qwen1.5-0.5B-Chat');
const prompt = "Return a single word that captures the main subject matter of the following passage. Make sure your answer a single word and nothing else: ";

// Create a feature extraction pipeline
const embedder = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1');

function normalizeCoordinates(pcaCoordinates) {
    // Find min and max for both axes
    const xValues = pcaCoordinates.map(coord => coord[0]);
    const yValues = pcaCoordinates.map(coord => coord[1]);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // Normalize coordinates to [0, 1]
    return pcaCoordinates.map(([x, y]) => [
        (x - xMin) / (xMax - xMin),
        (y - yMin) / (yMax - yMin)
    ]);
}

function toScreenCoordinates(normalizedCoordinates, screenWidth, screenHeight, padding = 40) { // Increased padding value
    // Define a drawable area reduced by padding
    const drawableWidth = screenWidth - 2 * padding;
    const drawableHeight = screenHeight - 2 * padding;

    // Apply further scaling to ensure no element is right on the edge
    const scaleMargin = padding; // Further reduce drawable area on each side

    return normalizedCoordinates.map(([x, y]) => {
        // Scale x and y to the drawable area with margins considered
        const left = (x * (drawableWidth - 2 * scaleMargin)) + padding + scaleMargin;
        const top = (y * (drawableHeight - 2 * scaleMargin)) + padding + scaleMargin;

        return { left, top };
    });
}

export default function App() {
    const [passage, setPassage] = useState('');
    const [summaries, setSummaries] = useState([]);
    const [embeddings, setEmbeddings] = useState([]);
    const [summaryCoords, setSummaryCoords] = useState([]);
    const [clickCount, setClickCount] = useState(0);
    const inputRef = useRef();

    const handleButtonClick = () => {
        setPassage(inputRef.current.value);
        setClickCount(clickCount + 1);
    };

    useEffect(() => {
        async function processInput() {
            if (!passage) return; // Exit early if there's no passage to process
    
            const text = generator.tokenizer.apply_chat_template([
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": prompt + passage }
            ], {
                tokenize: false,
                add_generation_prompt: true,
            });
    
            const output = await generator(text, {
                max_new_tokens: 128,
                do_sample: false,
            });
    
            const response = output[0].generated_text.split(/\r?\n/).pop().split(" ").pop().replace(/['".]+/g, '');
    
            // Check if lastSummary is already processed to avoid re-processing
            if (summaries.length > 0 && summaries[summaries.length - 1] === response) {
                return;
            }
    
            const embedding = await embedder(response, { pooling: 'mean', normalize: true });
            const newEmbeddings = [...embeddings, embedding["data"]]; // Temporary variable to hold new state
    
            const twoDimArrays = newEmbeddings.map(d => Array.from(d));
            const fitPCA = new PCA(twoDimArrays);
            const pcaEmbeddings = fitPCA.predict(twoDimArrays)["data"].map(d => Array.from(d.slice(0, 2)));
    
            const normalizedCoordinates = normalizeCoordinates(pcaEmbeddings);
            const screenCoords = toScreenCoordinates(normalizedCoordinates, window.innerWidth, window.innerHeight);
    
            // Update states together to ensure they are synchronized
            setSummaries(prevSummaries => [...prevSummaries, response]);
            setEmbeddings(newEmbeddings); 
            setSummaryCoords(screenCoords); 
        }
    
        processInput();
    }, [clickCount, passage]);
    
   
    return (
        <div>
            <input ref={inputRef} />
            <button onClick={handleButtonClick}>SUMMARIZE</button>
            <div className='responseBlock'>
                {summaryCoords[0]?.left ? 
                summaries.map((s, i) => (
                    <div 
                        key={i} 
                        className='summary' 
                        style={{
                            position: 'fixed',
                            left: `${summaryCoords[i].left}px`,
                            top: `${summaryCoords[i].top}px`
                        }}
                    >
                        {s}
                    </div>
                )) : 
                summaries.map((s, i) => (
                    <div 
                        key={i} 
                        className='summary' 
                        style={{
                            position: 'fixed',
                            left: '40%',
                            top: '40%'
                        }}
                    >
                        {s}
                    </div>
                ))
            }
            </div>
        </div>
    );
}
