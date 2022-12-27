"use client"

import Link from 'next/link';
import React from 'react';
import {auth} from '../../utils/firebase';
import {GoogleAuthProvider, signInWithPopup} from 'firebase/auth';

const Auth = () => {
	const googleAuth = new GoogleAuthProvider();

	const google = () => {
		try {
			signInWithPopup(auth, googleAuth)
			.then(async (result) => {
				console.log(result);
				const user = result.user;
				const data = user;

				const send = {
					uid: user.uid,
					token: await user.getIdToken()
				}
			}, (error) => {
				
			});
		} catch {

		}
	}

	return ( <div className='flex flex-col items-center w-full h-[100vh] justify-center'>
		<div className='text-3xl font-bold mb-3'>Login</div>
		<div className='text-2xl font-medium bg-white rounded-md flex flex-row p-3 gap-2' onClick={google}> <img src="/assets/google.png" /> Sign in using Google</div>
	</div>
  )
}

export default Auth;