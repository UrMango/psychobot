"use client"

import React, { useState } from 'react';

const Dashboard = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [reliabilities, setReliabilities] = useState<any>();

  const click = async () => {
    setResult(0);
    setReliabilities(null);
    const res = await fetch("http://localhost:8080/sentiment?sentence=" + text);
    const json = await res.json();
    if(json?.res) {
      let reliabilities:any = [];
      Object.keys(json.res?.reliability).forEach((key) => {
        console.log(key, json.res?.reliability[key])
        reliabilities.push(<p>{key}: {json.res?.reliability[key]}</p>)
      });
      setReliabilities(reliabilities);

      setResult(json.res);

    }
  }

  return (<div className='flex flex-col items-center'>
    <p>{text}</p>
	  <h1 className='text-2xl'>Dashboard</h1>
    <input placeholder='Text' onChange={(e) => setText(e.target.value)}/>
    <button onClick={click} >click me</button>

    <div className='bg-[rgba(0,0,0,0.5)] p-2 text-white'>
      {result != null && result != 0 && <>
        Results:
        <p>Sentence: {result?.sentence}</p>
        <p>Feeling: {result?.feeling}</p>
        <p>Reliability:</p>
        {reliabilities}
        </>} 
      {result == 0 && <p>Loading...</p>}
      {result == null && <p>Results will be here</p>}
    </div>
  </div>
  )
}

export default Dashboard