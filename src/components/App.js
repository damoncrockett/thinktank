import React, { useState, useRef, useEffect } from 'react';
import { pipeline } from '@xenova/transformers';

const generator = await pipeline('text-generation', 'Xenova/Qwen1.5-0.5B-Chat');
const prompt = "Return a single word that captures the content of the following passage. Do not return anything else: ";

export default function App() {
    const [passage, setPassage] = useState('');
    const [summary, setSummary] = useState('');
    const [clickCount, setClickCount] = useState(0);
    const inputRef = useRef();
    
    const handleButtonClick = () => {
        setPassage(inputRef.current.value);
        setClickCount(clickCount + 1);
    };

    useEffect(() => {
        async function getSummary() {
            if (passage) {
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
                console.log(output);
                setSummary(output[0].generated_text.split(/\r?\n/).pop()); // Get the last line; returns prompts otherwise
            }
        }
        getSummary();
    }, [clickCount]);
   
    return (
        <div>
            <input ref={inputRef} />
            <button onClick={handleButtonClick}>SUMMARIZE</button>
            <div className='responseBlock'>
                <div className='summary'>
                    {summary}
                </div>
            </div>
        </div>
    );
}
