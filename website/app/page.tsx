"use client"

import Link from 'next/link';
import React from 'react';

const Home = () => {
  const isDark = localStorage.getItem("mode") == "Dark";

  return (
  <div className="w-full h-full flex flex-col justify-center items-center" style={isDark ? {backgroundColor: "black", color: "white"} : {}}>
    <div className='w-full h-3/4 flex flex-col justify-center items-start'>
      <video className='absolute right-0 w-[60vw] h-3/4 object-cover object-left top-0 bottom-0 z-0' src="/assets/space-5200.mp4" autoPlay muted loop></video>
      <div className='flex flex-col justify-center items-center absolute z-10 '>
        <div className='ml-28'>
          <h1 className='text-5xl font-black text-left z-20'>A New Powerful AI</h1>
          <h1 className='text-4xl font-bold text-left z-20'>Detects your feelings with detail.</h1>
        </div>
        <Link href="/dashboard" className='text-2xl mt-5 font-medium bg-blue-600 rounded-full p-3 text-white'>Get Started</Link>
      </div>
    </div>
    <div className='w-full flex items-center justify-center'>
      <div className='w-4/5 text-lg'>
        <p>Revolutionary feeling analysis tool that uses advanced artificial intelligence to understand how you and others feel. Whether you want to improve your communication skills, enhance your emotional intelligence, or conduct research on human emotions, Psychobot AI can help you achieve your goals.</p>
      </div>
    </div>
  </div>
  )
}

export default Home;