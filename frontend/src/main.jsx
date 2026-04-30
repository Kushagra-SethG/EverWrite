import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, errorMessage: error?.message || 'Unknown UI error' };
  }

  componentDidCatch(error, errorInfo) {
    // Keep the app visible and log details for debugging.
    console.error('UI crash captured by AppErrorBoundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="ui-error-screen">
          <h1>The UI hit an error, but your game session is safe.</h1>
          <p>{this.state.errorMessage}</p>
          <button type="button" onClick={() => window.location.reload()}>Reload Interface</button>
        </div>
      );
    }

    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </React.StrictMode>,
);
