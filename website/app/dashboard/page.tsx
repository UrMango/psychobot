"use client"

import Link from 'next/link';
import React, { useState } from 'react';
import DashBox from './(components)/DashBox';
import QuickRun from './(components)/QuickRun';
import Result from './(components)/Results';

const Dashboard = () => {
  const isDark = localStorage.getItem("mode") == "Dark";

  const [result, setResult] = useState<any>(null);
  const [reliabilities, setReliabilities] = useState<any>();
  const [currentArch, setCurrentArch] = useState<string>("");

  const quickRun = async (input : string) => {
    setResult(0);
    setReliabilities(null);
    const res = await fetch("http://localhost:8080/sentiment?sentence=" + input);
    const json = await res.json();
    if(json?.res) {
      let reliabilities:any = [];

      if(json.res.reliability?.anger && json.res.reliability?.happy && json.res.reliability?.sadness) {
        const switches : any = {
          anger: "Happy",
          sadness: "Anger",
          happy: "Sadness"
        };
        console.log(json.res);
        json.res.feeling = switches[json.res.feeling];
        const happy = json.res.reliability?.anger;
        json.res.reliability.anger = json.res.reliability.sadness;
        json.res.reliability.sadness = json.res.reliability.happy;
        json.res.reliability.happy = happy;
      }
      
      Object.keys(json.res?.reliability).forEach((key) => {
        console.log(key, json.res?.reliability[key])
        const precent = (json.res?.reliability[key] * 100).toFixed(2);

        reliabilities.push(
          <div className='flex flex-col items-center justify-center'>
            <div style={{background: "linear-gradient(#00000000 " + Math.round(100 - Number(precent)).toString() + "%, #ff8426 " + Math.round(100 - Number(precent)).toString() + "%)"}} className='rounded-full w-[5rem] h-[5rem] border-solid border-2 border-white flex justify-center items-center'>
              <div className='text-center'>{precent}%</div>
            </div>
            <p className='text-center font-semibold'>{key}</p>
          </div>
        )
      });
      setReliabilities(reliabilities);
      setResult(json.res);
    }
  }
  
  return (
    <div className='w-full h-full flex items-center justify-center flex-col gap-10' style={isDark ? {backgroundColor: "black", color: "white"} : {}}>
      <div>
        <h1 className='text-3xl font-extrabold'>Your Dashboard</h1>
      </div>
      <div className="w-full flex items-center justify-center flex-row gap-8">
        <div className='w-1/3 flex flex-col gap-3'>
          <QuickRun isDark={isDark} onClick={quickRun}/>
          {result != null && 
            <Result reliabilities={reliabilities} result={result} />
          }
        </div>
        <div className='w-1/3 flex flex-row gap-3 justify-center items-center'>
          <DashBox text='Playground' link='/dashboard/playground' color1='#dd5e89' color2='#f7bb97' />
          <DashBox text='Coming soon...' link='/dashboard/' color1='#006df0' color2='#ad70e9' />
          {/* <h1 className='text-3xl font-bold'>Coming soon...</h1> */}
        </div>
      </div>
    </div>
  )
}

export default Dashboard;