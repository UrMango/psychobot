import Link from 'next/link';
import React from 'react';

const ProfileEdit = () => {
  return ( 
  <div className='w-full flex justify-center items-center flex-col gap-3'>
	  <h2 className='text-3xl font-bold'>Edit Profile</h2>
    <div className='flex justify-center gap-3'>
      <input type="text" placeholder='Username' />
      <input type="text" placeholder='Email' />
    </div>
    <div className='flex gap-2'>
      <label>Save history</label>
      <select>
        <option>Yes</option>
        <option>No</option>
      </select>
    </div>
    <button className='bg-white'>
      Update
    </button>
  </div>
  )
}

export default ProfileEdit;