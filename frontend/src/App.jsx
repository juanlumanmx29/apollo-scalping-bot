import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [botStatus, setBotStatus] = useState('stopped')
  const [trades, setTrades] = useState([])
  const [config, setConfig] = useState({
    symbol: 'BTCUSDT',
    quantity: 0.001,
    stopLoss: 0.02,
    takeProfit: 0.03
  })

  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

  useEffect(() => {
    fetchTrades()
  }, [])

  const fetchTrades = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/trades`)
      const data = await response.json()
      setTrades(data)
    } catch (error) {
      console.error('Error fetching trades:', error)
    }
  }

  const startBot = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      if (response.ok) {
        setBotStatus('running')
      }
    } catch (error) {
      console.error('Error starting bot:', error)
    }
  }

  const stopBot = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/stop`, { method: 'POST' })
      if (response.ok) {
        setBotStatus('stopped')
      }
    } catch (error) {
      console.error('Error stopping bot:', error)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚀 Apollo Scalping Bot</h1>
        <div className="status">
          Status: <span className={`status-${botStatus}`}>{botStatus}</span>
        </div>
      </header>

      <main>
        <div className="controls">
          <div className="config">
            <h2>Configuration</h2>
            <div className="form-group">
              <label>Symbol:</label>
              <input
                type="text"
                value={config.symbol}
                onChange={(e) => setConfig({...config, symbol: e.target.value})}
              />
            </div>
            <div className="form-group">
              <label>Quantity:</label>
              <input
                type="number"
                step="0.001"
                value={config.quantity}
                onChange={(e) => setConfig({...config, quantity: parseFloat(e.target.value)})}
              />
            </div>
            <div className="form-group">
              <label>Stop Loss (%):</label>
              <input
                type="number"
                step="0.01"
                value={config.stopLoss}
                onChange={(e) => setConfig({...config, stopLoss: parseFloat(e.target.value)})}
              />
            </div>
            <div className="form-group">
              <label>Take Profit (%):</label>
              <input
                type="number"
                step="0.01"
                value={config.takeProfit}
                onChange={(e) => setConfig({...config, takeProfit: parseFloat(e.target.value)})}
              />
            </div>
          </div>

          <div className="buttons">
            <button 
              onClick={startBot} 
              disabled={botStatus === 'running'}
              className="start-btn"
            >
              Start Bot
            </button>
            <button 
              onClick={stopBot} 
              disabled={botStatus === 'stopped'}
              className="stop-btn"
            >
              Stop Bot
            </button>
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