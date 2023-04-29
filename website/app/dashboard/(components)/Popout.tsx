import Link from "next/link";
import { useState } from "react";

const Popout = ({ title, text, buttonText } : { title : string, text : string, buttonText : string }) => {
	return (
		<div className="w-1/2 h-1/4 bg-red">
            <h1>{title}</h1>
            <h2>{text}</h1>
            <button>{buttonText}</button>
        </div>
	)
};

export default Popout;