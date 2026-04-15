import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// ========== 工具函数 ==========

// 生成 Session ID
const generateSessionId = () => {
  if (!localStorage.getItem('sessionId')) {
    localStorage.setItem('sessionId', `session_${Date.now()}`);
  }
  return localStorage.getItem('sessionId');
};

// Markdown 简单渲染（支持加粗、斜体、代码块）
const renderMarkdown = (text) => {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/【(.*?)】/g, '<span class="highlight-bracket">【$1】</span>');
};

// 打字机效果
const TypeWriter = ({ text, speed = 30 }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (displayedText.length < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(text.slice(0, displayedText.length + 1));
      }, speed);
      return () => clearTimeout(timer);
    } else {
      setIsComplete(true);
    }
  }, [displayedText, text, speed]);

  return <div dangerouslySetInnerHTML={{ __html: renderMarkdown(displayedText) }} />;
};

// ========== 主应用 ==========
export default function DontStarveChatBot() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      type: 'bot',
      content: '👋 欢迎来到《饥荒 Don\'t Starve》游戏攻略助手！\n\n我是一位资深玩家，精通所有生存策略、食物机制、季节过渡等内容。\n\n🎮 **你可以问我：**\n- 新手前期如何生存？\n- 理智值怎么管理？\n- 第几天应该造什么建筑？\n- 如何度过冬季？\n- 某个生物怎么对付？\n\n开始提问吧！',
      sources: [],
      apis: [],
      timestamp: new Date()
    }
  ]);

  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const sessionId = generateSessionId();

  // 自动滚动到最后
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 发送消息
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: messages.length,
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages([...messages, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputValue,
          session_id: sessionId
        })
      });

      if (!response.ok) throw new Error('网络错误');

      const data = await response.json();

      const botMessage = {
        id: messages.length + 1,
        type: 'bot',
        content: data.response,
        sources: data.sources || [],
        apis: data.apis_used || [],
        intent: data.intent || [],
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: messages.length + 1,
        type: 'bot',
        content: `❌ 连接失败: ${error.message}\n请确保后端服务正在运行 (localhost:5000)`,
        sources: [],
        apis: [],
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // 清空对话
  const handleClearChat = () => {
    if (window.confirm('确定要清空对话记录吗？')) {
      setMessages([messages[0]]);
      localStorage.removeItem('sessionId');
    }
  };

  // 处理回车发送
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="app-container">
      {/* 头部 */}
      <div className="app-header">
        <div className="header-content">
          <h1>🎮 饥荒游戏攻略助手</h1>
          <p>基于深度攻略文档 + 大模型的精准回答</p>
        </div>
        <button className="clear-btn" onClick={handleClearChat} title="清空对话">
          🗑️
        </button>
      </div>

      {/* 消息区域 */}
      <div className="messages-container">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-wrapper message-${msg.type}`}>
            <div className={`message-content message-${msg.type}`}>
              {msg.type === 'user' ? (
                <div className="user-message">{msg.content}</div>
              ) : (
                <div className="bot-message">
                  <TypeWriter text={msg.content} speed={20} />
                </div>
              )}

              {/* 信息来源展示 */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <div className="sources-label">📚 信息来源：</div>
                  {msg.sources.map((source, idx) => (
                    <span key={idx} className="source-tag">
                      {source}
                    </span>
                  ))}
                </div>
              )}

              {/* API 调用标记 */}
              {msg.apis && msg.apis.length > 0 && (
                <div className="message-apis">
                  <span className="api-tag">🔗 APIs: {msg.apis.join(', ')}</span>
                </div>
              )}

              {/* 意图识别标记 */}
              {msg.intent && msg.intent.length > 0 && (
                <div className="message-intent">
                  <span className="intent-tag">🎯 识别: {msg.intent.join(' + ')}</span>
                </div>
              )}

              <span className="message-time">
                {msg.timestamp.toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message-wrapper message-bot">
            <div className="message-content message-bot loading">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <span>正在查阅知识库...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入你的问题...（Shift+Enter 换行，Enter 发送）"
            disabled={isLoading}
            rows="3"
            className="message-input"
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !inputValue.trim()}
            className="send-btn"
            title="发送 (Enter)"
          >
            {isLoading ? '⏳' : '发送'}
          </button>
        </div>
        <div className="input-hints">
          <span>💡 提示：询问具体的生存策略、建筑顺序、季节过渡等内容</span>
        </div>
      </div>
    </div>
  );
}