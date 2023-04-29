import { useState } from "react";

const QuickRun = ({ onClick, isDark } : { onClick : any, isDark : boolean }) => {
	const [input, setInput] = useState("");

	return (
		<div className='rounded-3xl w-full p-3 flex flex-col gap-1' style={isDark ? {color: "white", backgroundColor: "black", border: "2px solid white"} : {border: "2px solid black"}}>
		<p className='font-semibold text-lg'>Quick Run</p>
		<div className='w-full flex flex-row gap-2'>
			<input className='rounded-md p-2 w-full' style={isDark ? {color: "white", backgroundColor: "black", border: "1px solid white"} : {border: "1px solid black"}} placeholder="I'm so depressed... my life is rough and hard" onChange={(e) => setInput(e.target.value)}/>
			<button onClick={() => onClick(input)} className='rounded-md p-2' style={isDark ? {color: "white", backgroundColor: "black", border: "4px solid white"} : {border: "4px solid black"}}>
				Run
			</button>
		</div>
		</div>
	)
};

export default QuickRun;