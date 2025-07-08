import { useState, useEffect } from 'react'
import './App.css'
import { useAuth } from './AuthContext'

function App() {
  const [botStatus, setBotStatus] = useState('stopped')
  const [trades, setTrades] = useState([])
  const [logs, setLogs] = useState([])
  const { user, token, loading, signIn, signOut } = useAuth()

  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'https://apollo-scalping-bot-backend.onrender.com'

  useEffect(() => {
    if (user && token) {
      fetchTrades()
      fetchLogs()
      checkBotStatus()
      // Poll for updates every 5 seconds
      const interval = setInterval(() => {
        fetchTrades()
        fetchLogs()
        checkBotStatus()
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [user, token])

  const fetchTrades = async () => {
    try {
      console.log('Fetching trades with token:', token?.substring(0, 50) + '...')
      const response = await fetch(`${backendUrl}/trades`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        console.error('API Error:', response.status, errorData)
        throw new Error(`API Error: ${response.status}`)
      }
      
      const data = await response.json()
      setTrades(data.trades || [])
    } catch (error) {
      console.error('Error fetching trades:', error)
      setTrades([])
    }
  }

  const fetchLogs = async () => {
    try {
      const response = await fetch(`${backendUrl}/paper-trading-logs`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (response.ok) {
        const data = await response.json()
        setLogs(data.logs || [])
      }
    } catch (error) {
      console.error('Error fetching logs:', error)
    }
  }

  const checkBotStatus = async () => {
    try {
      const response = await fetch(`${backendUrl}/paper-trading-status`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (response.ok) {
        const data = await response.json()
        // Convert backend status to frontend status
        if (data.status === "activo") {
          setBotStatus('running')
        } else {
          setBotStatus('stopped')
        }
      }
    } catch (error) {
      console.error('Error checking bot status:', error)
    }
  }

  const startBot = async () => {
    try {
      console.log('Starting paper trading...')
      const response = await fetch(`${backendUrl}/start-paper-trading`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (response.ok) {
        const data = await response.json()
        console.log('Start response:', data)
        setBotStatus('running')
        fetchLogs() // Immediately fetch logs
        checkBotStatus() // Double-check status
      } else {
        console.error('Failed to start bot:', response.status)
      }
    } catch (error) {
      console.error('Error starting bot:', error)
    }
  }

  const stopBot = async () => {
    try {
      console.log('Stopping paper trading...')
      const response = await fetch(`${backendUrl}/stop-paper-trading`, { 
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (response.ok) {
        const data = await response.json()
        console.log('Stop response:', data)
        setBotStatus('stopped')
        checkBotStatus() // Double-check status
      } else {
        console.error('Failed to stop bot:', response.status)
      }
    } catch (error) {
      console.error('Error stopping bot:', error)
    }
  }

  const handleLogin = async () => {
    try {
      await signIn()
    } catch (error) {
      console.error('Login failed:', error)
    }
  }

  const handleLogout = async () => {
    try {
      await signOut()
      setBotStatus('stopped')
      setTrades([])
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }

  if (loading) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>🚀 Apollo Scalping Bot</h1>
          <p>Loading...</p>
        </header>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>🚀 Apollo Scalping Bot</h1>
          <p>Please sign in to continue</p>
        </header>
        <main>
          <div className="login-form">
            <h2>Welcome to Apollo</h2>
            <p>Sign in with your Google account to access the trading bot</p>
            <button onClick={handleLogin} className="google-signin-btn">
              <svg width="18" height="18" viewBox="0 0 18 18" style={{ marginRight: '10px' }}>
                <path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18Z"/>
                <path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2.04a4.8 4.8 0 0 1-2.7.75 4.8 4.8 0 0 1-4.52-3.4H1.83v2.07A8 8 0 0 0 8.98 17Z"/>
                <path fill="#FBBC05" d="M4.46 10.37a4.8 4.8 0 0 1 0-2.74V5.56H1.83a8 8 0 0 0 0 6.81l2.63-2.07Z"/>
                <path fill="#EA4335" d="M8.98 4.23c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 8.98 1a8 8 0 0 0-7.15 4.56l2.63 2.07c.61-1.8 2.26-3.4 4.52-3.4Z"/>
              </svg>
              Sign in with Google
            </button>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚀 Apollo Scalping Bot</h1>
        <div className="header-info">
          <div className="user-info">
            <img src={user.photoURL} alt="Profile" className="profile-pic" />
            <span>Welcome, {user.displayName}</span>
          </div>
          <div className="status">
            Status: <span className={`status-${botStatus}`}>{botStatus}</span>
          </div>
          <button onClick={handleLogout} className="logout-btn">
            Sign Out
          </button>
        </div>
      </header>

      <main>
        <div className="controls">
          <div className="config">
            <h2>Trading Configuration</h2>
            <div className="config-info">
              <div className="config-item">
                <label>Symbol:</label>
                <span>ETH/USDT</span>
              </div>
              <div className="config-item">
                <label>Capital:</label>
                <span>$10,000</span>
              </div>
              <div className="config-item">
                <label>Min Sell:</label>
                <span>+0.16%</span>
              </div>
              <div className="config-item">
                <label>Stop Loss:</label>
                <span>-1.0%</span>
              </div>
              <div className="config-item">
                <label>Trailing Stop:</label>
                <span>0.2% below max</span>
              </div>
              <div className="config-item">
                <label>Binance Commission:</label>
                <span>0.075% x2</span>
              </div>
              <div className="config-item">
                <label>ML Threshold:</label>
                <span>0.73</span>
              </div>
              <div className="config-item">
                <label>Strategy:</label>
                <span>Apollo Scalping</span>
              </div>
            </div>
          </div>

          <div className="buttons">
            <button 
              onClick={startBot} 
              disabled={botStatus === 'running'}
              className="start-btn"
            >
              Start Paper Trading
            </button>
            <button 
              onClick={stopBot} 
              disabled={botStatus === 'stopped'}
              className="stop-btn"
            >
              Stop Paper Trading
            </button>
          </div>
        </div>

        <div className="logs">
          <h2>Live Trading Logs</h2>
          <div className="logs-container">
            {logs.length === 0 ? (
              <p>No trading activity yet. Start the bot to see live logs.</p>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="log-item">
                  {log}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="trades">
          <h2>Recent Trades</h2>
          <div className="trades-list">
            {trades.length === 0 ? (
              <p>No trades yet</p>
            ) : (
              trades.map((trade, index) => (
                <div key={index} className="trade-item">
                  <span className="trade-symbol">{trade.symbol}</span>
                  <span className="trade-side">{trade.side}</span>
                  <span className="trade-quantity">{trade.quantity}</span>
                  <span className="trade-price">{trade.price}</span>
                  <span className="trade-time">{new Date(trade.time).toLocaleString()}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App