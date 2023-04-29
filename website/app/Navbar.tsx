"use client"

import Link from "next/link";
import { useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";

const Navbar = () => {
	const [darkLight, setDarkLight] = useState("Dark");
	let [isDark, setIsDark] = useState(localStorage.getItem("mode") == "Dark");
	const router = useRouter();

	useEffect(() => {
		let mode = localStorage.getItem("mode");
		if(mode == "Dark") {
			setDarkLight("Light");
		}
	}, []);

	const switchDarkLight = () => {
		let mode = localStorage.getItem("mode");
		if(mode == null) {
			localStorage.setItem("mode", "Dark");
			mode = localStorage.getItem("mode");
		}
		if(mode == "Dark") {
			localStorage.setItem("mode", "Light");
			setDarkLight("Dark");
			router.refresh();
		} else {
			localStorage.setItem("mode", "Dark");
			setDarkLight("Light");
			router.refresh();
		}
		setTimeout(() => {
			setIsDark(localStorage.getItem("mode") == "Dark");
		}, 200)
	};


	return (
		<div className="fixed w-full h-full pointer-events-none z-50" style={isDark ? {color: "white"} : {}}>
			<div className="flex flex-col items-center justify-center py-6 pointer-events-auto">
				<h3 className="font-black text-3xl">PsychoBot AI</h3>
				<div className="w-1/2 flex items-center justify-center gap-4 font-medium text-xl" >
					<Link href="/">Home</Link>
					<Link href="/dashboard">Dashboard</Link>
					<h2 onClick={switchDarkLight} className="cursor-pointer">{darkLight} Mode</h2>
				</div>
			</div>
			{/* <div className="absolute w-full bottom-0 py-6 flex flex-row items-center gap-4 justify-center font-normal text-xl pointer-events-auto" style={isDark ? {borderTop: "2px solid white", backgroundColor: "black"} : {borderTop: "2px solid black", backgroundColor: "white"}}>
				<Link href="/dashboard/playground" className="rounded-full p-3" style={isDark ? {color: "white", backgroundColor: "black", border: "2px solid white"} : {border: "2px solid black"}}>Playground</Link>
				<Link href="/dashboard/train" className="rounded-full p-3" style={isDark ? {color: "white", backgroundColor: "black", border: "2px solid white"} : {border: "2px solid black"}}>Train Model</Link>
			</div> */}
		</div>)
};

export default Navbar;