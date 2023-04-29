import Link from "next/link";
import { useState } from "react";

const DashBox = ({ text, link, color1, color2 } : { text : string, link : string, color1 : string, color2 : string }) => {
	return (
		<Link href={link} style={{background: "linear-gradient(" + color1 + " 0%, " + color2 + " 100%)"}} 
			className='rounded-3xl w-full p-5 flex flex-col gap-1 text-white w-[20rem] h-[10rem] justify-center items-center'>
			<h2 className="text-xl">{text}</h2>
		</Link>
	)
};

export default DashBox;