import { useState } from "react";

const Result = ({ result, reliabilities } : { result : any, reliabilities : any }) => {

	return (
		<div className='bg-gradient-to-r from-[#8547c5] to-[#aa5c8d] rounded-3xl w-full p-5 flex flex-col gap-1 text-white'>
          <div className='w-full flex items-center justify-center'>
            {result == 0 && <p className='font-medium'>Analyzing your text...</p>}
          </div>
          {
            result != 0 && result != null && 
            <div className='w-full flex flex-col justify-center items-center gap-2'>
              <p className='w-full text-left font-semibold text-lg'>Quick Run Results</p>
              <p className=''>"{result?.sentence}"</p>
              <p className='font-bold text-xl'>🎉 {result?.feeling} 🎉</p>
              <div className='flex flex-row gap-2'>
                {reliabilities}
              </div>
            </div>
          }
        </div>
	)
};

export default Result;