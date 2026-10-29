import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ error, errorInfo });
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '24px', textAlign: 'center', background: '#030b14', color: '#dcf7ff', minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontFamily: 'Rajdhani, sans-serif' }}>
          <h1 style={{ color: '#ff718d' }}>Something went wrong.</h1>
          <pre style={{ background: '#04111c', padding: '10px', borderRadius: '4px', overflowX: 'auto', maxWidth: '90vw', marginTop: '10px', textAlign: 'left' }}>
            {this.state.error && this.state.error.toString()}
          </pre>
          {this.state.errorInfo && (
            <details style={{ marginTop: '10px', textAlign: 'left' }}>
              <summary>Stack details</summary>
              <pre>{this.state.errorInfo.componentStack}</pre>
            </details>
          )}
          <button
            onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
            style={{ marginTop: '20px', padding: '10px 24px', background: '#27d9ff', color: '#030b14', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
