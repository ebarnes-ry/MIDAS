import React from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { FullVisionPipeline } from './components/vision/FullVisionPipeline';
import './App.css';

function App() {
  return (
    <ErrorBoundary>
      <FullVisionPipeline />
    </ErrorBoundary>
  );
}

export default App;
