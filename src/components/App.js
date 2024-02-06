import React, { useState, useRef } from 'react';

export default function App() {
    const [name, setName] = useState('');
    const inputRef = useRef();
    
    const handleButtonClick = () => {
        setName(inputRef.current.value);
    };
    
    return (
        <div>
        <input ref={inputRef} />
        <button onClick={handleButtonClick}>Set Name</button>
        <p>Hello, {name}</p>
        </div>
    );
    }

    