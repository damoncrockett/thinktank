import React, { useState, useRef, useEffect } from 'react';
import { pipeline } from '@xenova/transformers';
import { PCA } from 'ml-pca';
import { normalizeCoordinates, toScreenCoordinates } from '../../utils/coords';
import { select } from 'd3-selection';

const generator = await pipeline('text-generation', 'Xenova/Qwen1.5-0.5B-Chat');
const embedder = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1');
const prompt = "Return a single word that captures the main subject matter of the following passage. Make sure your answer a single word and nothing else: ";

export default function App() {
    const [passage, setPassage] = useState('');
    const [summaries, setSummaries] = useState([]);
    const [embeddings, setEmbeddings] = useState([]);
    const [summaryCoords, setSummaryCoords] = useState([]);
    const inputRef = useRef();
    const svgRef = useRef();

    function handleKeyDown(event) {
        if (event.key === 'Enter') {
            handleButtonClick(); // Call the function to handle submission
        }
    }

    const handleButtonClick = () => {
        setPassage(inputRef.current.value);
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
            const screenCoords = toScreenCoordinates(normalizedCoordinates, svgRef.current.clientWidth, svgRef.current.clientHeight, 100);
    
            // Update states together to ensure they are synchronized
            setSummaries(prevSummaries => [...prevSummaries, response]);
            setEmbeddings(newEmbeddings); 
            setSummaryCoords(screenCoords); 
        }
    
        processInput();
    }, [passage]);

    useEffect(() => {
        if ( summaryCoords.length === 0 ) return;
    
        const svg = select(svgRef.current);
        svg.selectAll("*").remove(); // Clear previous SVG elements
    
        const padding = 10; // Padding inside the rectangle
    
        // Append groups for each summary
        const summariesGroup = svg.selectAll('g')
            .data(summaryCoords)
            .enter()
            .append('g'); // Group for each summary for easier positioning
    
        // First, append text elements permanently
        summariesGroup.append('text')
            .text((d, i) => summaries[i])
            .attr('x', d => d.left ? d.left + padding : 200 + padding)
            .attr('y', d => d.top ? d.top + 20 : 200 + 20) // Center text vertically
            .attr('dominant-baseline', 'middle')
            .style('font-size', '12px')
            .each(function(d, i) {
                const textWidth = this.getComputedTextLength();
                const rectWidth = textWidth + 2 * padding; // Calculate rectangle width based on text width
    
                // Append a rectangle for each text, inserting it before the text element
                select(this.parentNode).insert('rect', 'text')
                    .attr('x', d.left ? d.left : 200)
                    .attr('y', d.top ? d.top : 200)
                    .attr('width', rectWidth)
                    .attr('height', 40) // Fixed height, adjust as needed
                    .attr('rx', 10) // Rounded corners
                    .attr('ry', 10)
                    .style('fill', 'aquamarine');
            });

            inputRef.current.value = '';
    
    }, [summaryCoords]);
    
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
            <div style={{ marginBottom: '10px' }}> {/* Ensure this div doesn't grow and only takes necessary space */}
                <input ref={inputRef} onKeyDown={handleKeyDown} />
                <button onClick={handleButtonClick}>SUMMARIZE</button>
            </div>
            <svg ref={svgRef} style={{ flexGrow: 1 }} width="100%" height="100%"></svg>
        </div>
    );
    
}
