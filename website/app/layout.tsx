import React from 'react'
import "../styles/globals.css"
import Navbar from './Navbar'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html>
      <head />
      <body className='bg-slate-400'>
        {/* <Navbar /> */}
        {children}</body>
    </html>
  )
}
