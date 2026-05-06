/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
  ],
  theme: {
    extend: {
      colors: {
        ink:       { DEFAULT: '#1a1714', 2: '#3d3830', 3: '#7a7067' },
        rule:      { DEFAULT: '#d8d0c4', lt: '#ece7df' },
        parchment: '#f8f5ef',
        cream:     '#fdfaf5',
        verified:  { DEFAULT: '#2d5a3d', bg: '#eef5f1', bd: '#a8c9b4' },
        failed:    { DEFAULT: '#7a2020', bg: '#f9eeee', bd: '#c9a0a0' },
        accent:    { DEFAULT: '#1e3a5f', lt: '#e8eef5', mid: '#c4d3e8' },
        amber:     { DEFAULT: '#7a5c1e', bg: '#fdf5e6', bd: '#d4b870' },
      },
      fontFamily: {
        serif:  ["'EB Garamond'", 'Georgia', 'serif'],
        serif2: ["'Crimson Pro'", 'Georgia', 'serif'],
        mono:   ["'JetBrains Mono'", "'Courier New'", 'monospace'],
      },
      spacing: { '18': '4.5rem', '88': '22rem' },
      borderRadius: { 'xl': '0.75rem', '2xl': '1rem' },
    },
  },
  plugins: [],
}
