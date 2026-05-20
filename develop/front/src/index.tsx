import React from 'react';
import App from './App';
import './styles/globals.css';
import { createRoot } from 'react-dom/client';

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Root element not found');
const root = createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);