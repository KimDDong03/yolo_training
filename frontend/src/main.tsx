import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { DialogProvider } from './components/ui/Dialog'
import { ToastProvider } from './components/ui/Toast'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ToastProvider>
      <DialogProvider>
        <App />
      </DialogProvider>
    </ToastProvider>
  </React.StrictMode>,
)
