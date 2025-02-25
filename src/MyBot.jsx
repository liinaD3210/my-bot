import React, { useState } from 'react';

const SimpleBot = () => {
  // Состояние для хранения сообщений
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');

  // Функция для обработки отправки сообщения
  const handleSendMessage = async () => {
    if (userInput.trim()) {
      const newMessage = {
        text: userInput,
        sender: 'user',
      };

      // Отправка данных на бэкенд
      const response = await fetch('http://127.0.0.1:5000/bot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
      });
  
      const data = await response.json();
  
      if (response.ok) {
        // Ответ бота
        const botResponse = {
          text: data.response,
          sender: 'bot',
        };
        setMessages((prevMessages) => [...prevMessages, newMessage, botResponse]);
      } else {
        // Если ошибка
        const errorResponse = {
          text: 'Что-то пошло не так. Попробуйте еще раз.',
          sender: 'bot',
        };
        setMessages((prevMessages) => [...prevMessages, newMessage, errorResponse]);
      }
  
      setUserInput('');
    }
  };
  

  // Рендерим все сообщения
  const renderMessages = () => {
    return messages.map((message, index) => (
      <div key={index} style={{ textAlign: message.sender === 'user' ? 'right' : 'left' }}>
        <p style={{ display: 'inline-block', padding: '10px', borderRadius: '10px', backgroundColor: message.sender === 'user' ? '#e1ffc7' : '#f0f0f0' }}>
          {message.text}
        </p>
      </div>
    ));
  };

  return (
    <div style={{ width: '400px', margin: '0 auto', padding: '20px', border: '1px solid #ccc', borderRadius: '10px' }}>
      <div style={{ maxHeight: '300px', overflowY: 'auto', marginBottom: '20px' }}>
        {renderMessages()}
      </div>

      <div style={{ display: 'flex' }}>
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Введите сообщение..."
          style={{ flex: 1, padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }}
        />
        <button
          onClick={handleSendMessage}
          style={{ padding: '10px', marginLeft: '10px', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          Отправить
        </button>
      </div>
    </div>
  );
};

export default SimpleBot;
