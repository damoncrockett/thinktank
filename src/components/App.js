import React, { useState, useRef, useEffect } from 'react';
import { pipeline } from '@xenova/transformers';
import { PCA } from 'ml-pca';
import { normalizeCoordinates, toScreenCoordinates } from '../../utils/coords';
import { chooseThreeMostImportantWords } from '../../utils/text';
import { select } from 'd3-selection';
import { transition } from 'd3-transition';
import { forceSimulation, forceManyBody, forceX, forceY, forceCollide } from 'd3-force';

const embedder = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1');

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
        inputRef.current.value = ''; // Clear the input field
    };

    useEffect(() => {
        async function processInput() {
            if (!passage) return; // Exit early if there's no passage to process
           
            const embedding = await embedder(passage, { pooling: 'mean', normalize: true });
            const newEmbeddings = [...embeddings, embedding["data"]]; // Temporary variable to hold new state
    
            const twoDimArrays = newEmbeddings.map(d => Array.from(d));
            const fitPCA = new PCA(twoDimArrays);
            const pcaEmbeddings = fitPCA.predict(twoDimArrays)["data"].map(d => Array.from(d.slice(0, 2)));
    
            const normalizedCoordinates = normalizeCoordinates(pcaEmbeddings);
            const screenCoords = toScreenCoordinates(normalizedCoordinates, svgRef.current.clientWidth, svgRef.current.clientHeight, 100);
    
            // Update states together to ensure they are synchronized
            setSummaries(prevSummaries => [...prevSummaries, chooseThreeMostImportantWords(passage)]);
            setEmbeddings(newEmbeddings); 
            setSummaryCoords(screenCoords); 
        }
    
        processInput();
    }, [passage]);

    useEffect(() => {
        if (summaryCoords.length === 0) return;
    
        const svg = select(svgRef.current);

        const simulation = forceSimulation(summaryCoords)
            .force("charge", forceManyBody().strength(-10))
            .force("x", forceX(d => d.x).strength(0.5))
            .force("y", forceY(d => d.y).strength(0.5))
            .force("collide", forceCollide().radius(20).strength(0.5))
            .stop();

        // Manually run the simulation for a set number of ticks
        for (let i = 0; i < 10; i++) simulation.tick();

        // Transition duration in milliseconds
        const duration = 750;
    
        // Update pattern: Bind data to groups, handling entering (new) and updating (existing) elements
        let summariesGroup = svg.selectAll('g')
            .data(summaryCoords, (_, i) => i); // Assuming each data point has an identifier 'id'
    
        // Enter new elements
        const enterGroup = summariesGroup.enter()
            .append('g')
            .attr('transform', d => `translate(${d.left},${d.top})`); // Initial position
    
        enterGroup.append('rect')
            .attr('width', 100) // Initial width, adjust based on expected text width
            .attr('height', 40) // Fixed height
            .attr('rx', 20) // Rounded corners
            .attr('ry', 20)

        const padding = 10;
    
        enterGroup.append('text')
            .attr('x', padding)
            .attr('y', 20) // Center text vertically
            .attr('dominant-baseline', 'middle')
            .style('fill', 'black')
            .text((_, i) => summaries[i]);
    
        // Merge entering elements with updating ones to apply transitions
        summariesGroup = enterGroup.merge(summariesGroup);
    
        // Apply transitions to all groups (both new and updating)
        summariesGroup.transition().duration(duration)
            .attr('transform', d => `translate(${d.left},${d.top})`);
    
        summariesGroup.select('text')
            .each(function(d, i) {
                const textWidth = this.getComputedTextLength();
                const rectWidth = textWidth + 2 * padding;
                select(this.previousSibling) // Assuming the rectangle is immediately before the text in the DOM
                    .transition().duration(duration)
                    .attr('width', rectWidth);
            });
    
    }, [summaryCoords]); // Depend on summaries as well since they're used in rendering
    
    
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
