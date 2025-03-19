import React, { useState } from 'react';
import backgroundImage from './DALL·E-2025-03-19-21.51.jpg';

const SimpleBot = () => {
  // Состояние для хранения сообщений
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  // Состояние для отображения процесса «размышления»
  const [isLoading, setIsLoading] = useState(false);

  // Функция для обработки отправки сообщения
  const handleSendMessage = async () => {
    if (userInput.trim()) {
      // Добавляем сообщение пользователя
      const newMessage = {
        text: userInput,
        sender: 'user',
      };

      setMessages((prevMessages) => [...prevMessages, newMessage]);
      setUserInput('');
      // Включаем индикатор загрузки
      setIsLoading(true);

      try {
        // Отправляем запрос к бэкенду
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
          setMessages((prevMessages) => [...prevMessages, botResponse]);
        } else {
          // Если ошибка
          const errorResponse = {
            text: 'Что-то пошло не так. Попробуйте еще раз.',
            sender: 'bot',
          };
          setMessages((prevMessages) => [...prevMessages, errorResponse]);
        }
      } catch (error) {
        // В случае сбоя запроса
        const errorResponse = {
          text: 'Ошибка при запросе. Попробуйте еще раз позже.',
          sender: 'bot',
        };
        setMessages((prevMessages) => [...prevMessages, errorResponse]);
      } finally {
        // Выключаем индикатор загрузки
        setIsLoading(false);
      }
    }
  };

  // Рендерим все сообщения
  const renderMessages = () => {
    return messages.map((message, index) => (
      <div
        key={index}
        style={{
          textAlign: message.sender === 'user' ? 'right' : 'left',
          margin: '10px 0',
        }}
      >
        <p
          style={{
            display: 'inline-block',
            padding: '10px 15px',
            borderRadius: '15px',
            maxWidth: '70%',
            backgroundColor: message.sender === 'user' ? '#d8ffd8' : '#f0f0f0',
            whiteSpace: 'pre-wrap', // сохраняем абзацы и переносы строк
          }}
        >
          {message.text}
        </p>
      </div>
    ));
  };

  return (
    // Обёртка для фона
    <div
      style={{
        minHeight: '100vh', // фон на всю высоту экрана
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
      }}
    >
      {/* Блок с самим чатом */}
      <div
        style={{
          width: '400px',
          margin: '40px auto',
          padding: '20px',
          border: '1px solid #ccc',
          borderRadius: '10px',
          fontFamily: 'sans-serif',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          backgroundColor: '#fff', // фон для чата, чтобы текст не сливался с фоном
        }}
      >
        <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>Чат-менеджер кофейни</h2>

        <div
          style={{
            maxHeight: '300px',
            overflowY: 'auto',
            marginBottom: '20px',
            padding: '10px',
            backgroundColor: '#fafafa',
            borderRadius: '5px',
          }}
        >
          {renderMessages()}
          {isLoading && (
            <div style={{ textAlign: 'left', margin: '10px 0' }}>
              <p
                style={{
                  display: 'inline-block',
                  padding: '10px 15px',
                  borderRadius: '15px',
                  maxWidth: '70%',
                  backgroundColor: '#fff3cd',
                }}
              >
                Подождите, менеджер думает...
              </p>
            </div>
          )}
        </div>

        <div style={{ display: 'flex' }}>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Введите сообщение..."
            style={{
              flex: 1,
              padding: '10px',
              borderRadius: '5px',
              border: '1px solid #ccc',
              marginRight: '10px',
            }}
            onKeyDown={(e) => {
              // Отправляем сообщение по Enter
              if (e.key === 'Enter') {
                handleSendMessage();
              }
            }}
          />
          <button
            onClick={handleSendMessage}
            style={{
              padding: '10px 20px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
            }}
          >
            Отправить
          </button>
        </div>
      </div>
    </div>
  );
};

export default SimpleBot;
